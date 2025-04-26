package spark.kp4

import models.GBTModelTrainer
import preprocessor.FeaturePreprocessor

import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


class KafkaService(spark: SparkSession) {

  val model: GBTClassificationModel = GBTClassificationModel.load(GBTModelTrainer.modelPath)
  val featurePreprocessor = new FeaturePreprocessor()
  val kafkaBootstrapServers = "127.0.0.1:9094"
  val kafkaTopic = "transactions"

  def process(): Unit = {
    val kafkaStream: DataFrame = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("subscribe", kafkaTopic)
      .option("startingOffsets", "earliest")
      .load()

    val eventStream = kafkaStream
      .selectExpr("CAST(value AS STRING) as json")
      .select(from_json(col("json"), FeaturePreprocessor.schema).as("data"))
      .select("data.*")
      .transform(featurePreprocessor.assembleStreamingData)

    // Передбачення за допомогою GBT моделі
    val processedStream = model.transform(eventStream)

    // Перетворення результатів у JSON та перейменування стовпця на 'value'
    val outputStream = processedStream
      .select(to_json(struct("*")).as("value"))

    // Вивід результатів у консоль або запис у Kafka
    val query = outputStream
      .writeStream
      .format("kafka") // Можна змінити на 'kafka' для запису назад
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("topic", "transactions_processed")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    query.awaitTermination()
  }

  def uploadTestingDataToKafka(): Unit = {
    val delayMillis = 300

    val df = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .schema(FeaturePreprocessor.schema)
      .load("data/processed/test/")

    df.toJSON.collect().foreach { row =>
      Thread.sleep(delayMillis)
      val kafkaDF = spark.createDataFrame(java.util.Collections.singletonList(Row(row)), StructType(Array(StructField("value", StringType))))
      println(s"Sending message: $row")
      kafkaDF.write
        .format("kafka")
        .option("kafka.bootstrap.servers", kafkaBootstrapServers)
        .option("topic", kafkaTopic)
        .save()
    }

  }

}

object KafkaListenerApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Antifraud")
      .master("local[*]")
      .config("spark.sql.streaming.checkpointLocation", "checkpoint")
      .getOrCreate()

    val processor = new KafkaService(spark)
    processor.process()
  }
}

object KafkaProducerApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("KafkaProducer")
      .master("local[*]")
      .getOrCreate()

    val processor = new KafkaService(spark)
    processor.uploadTestingDataToKafka()
  }
}

