package spark.kp4

import metrics.MetricsService
import models.GBTModelTrainer
import preprocessor.FeaturePreprocessor

import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.elasticsearch.spark.sql._ // Import for Elasticsearch

class KafkaService(spark: SparkSession) {

  val model: GBTClassificationModel = GBTClassificationModel.load(GBTModelTrainer.modelPath)
  val featurePreprocessor = new FeaturePreprocessor()
  val kafkaBootstrapServers = "127.0.0.1:9094"
  val kafkaTopic = "transactions"

  def process(): Unit = {
    MetricsService.startServer()

    val kafkaStream: DataFrame = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("subscribe", kafkaTopic)
      .option("startingOffsets", "earliest")
      .option("failOnDataLoss", "false")
      .load()

    val eventStream = kafkaStream
      .selectExpr("CAST(value AS STRING) as json")
      .select(from_json(col("json"), FeaturePreprocessor.schema).as("data"))
      .select("data.*")
      .transform(featurePreprocessor.assembleStreamingData)

    val processedStream = model.transform(eventStream)

    val metricsStream = processedStream
      .withColumn("isFraud", when(col("prediction") === 1.0, true).otherwise(false))
      .withColumn("prob_array", vector_to_array(col("probability")))
      .withColumn("probability", col("prob_array").getItem(1))
      .withColumn("isHighRisk", when(col("probability") > 0.4, true).otherwise(false))
      .drop("prob_array")

    val metricsQuery = metricsStream.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        MetricsService.generateBatchMetrics(batchDF)
      }
      .start()

    val outputStream = metricsStream
      .select(to_json(struct("*")).as("value"))

    val query = outputStream
      .writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("topic", "transactions_processed")
      .option("checkpointLocation", "checkpoint-kafka") // Checkpoint for Kafka
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()

    // New Elasticsearch stream for metricsStream
    val esQuery = metricsStream // Writing the structured metricsStream
      .writeStream
      .format("org.elasticsearch.spark.sql")
      .option("es.nodes", "elasticsearch")
      .option("es.port", "9200")
      .option("es.resource", "transactions_processed/_doc") // Index and type
      .option("es.nodes.wan.only", "true")
      .option("checkpointLocation", "checkpoint-elasticsearch") // Checkpoint for Elasticsearch
      .trigger(Trigger.ProcessingTime("10 seconds")) // Match Kafka trigger
      .start()

    spark.streams.awaitAnyTermination() // Wait for all streams (Kafka and ES)
  }

  def uploadTestingDataToKafka(): Unit = {
    val df = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .schema(FeaturePreprocessor.schema)
      .load("data/processed/test/")

    df.toJSON.collect().foreach { row =>
      val randomDelay = Math.random() * 500
      Thread.sleep(randomDelay.toLong)
      val kafkaDF = spark.createDataFrame(java.util.Collections.singletonList(Row(row)), StructType(Array(StructField("value", StringType))))
      println(s"Sending message: $row")
      kafkaDF.write
        .format("kafka")
        .option("kafka.bootstrap.servers", kafkaBootstrapServers)
        .option("topic", kafkaTopic)
        .save()
    }
  }

  def shutdown(): Unit = {
    println("Initiating graceful shutdown of Kafka service...")

    println("Stopping metrics server...")
    MetricsService.stopServer()

    if (!spark.sparkContext.isStopped) {
      println("Stopping Spark session...")
      spark.stop()
    }

    println("Kafka service shutdown complete")
  }

}

object KafkaListenerApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Antifraud")
      .master("local[*]")
      // .config("spark.sql.streaming.checkpointLocation", "checkpoint") // Removed global checkpoint
      .getOrCreate()

    val processor = new KafkaService(spark)

    Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run(): Unit = {
        println("Shutdown hook triggered, performing graceful shutdown...")
        processor.shutdown()
      }
    })

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
