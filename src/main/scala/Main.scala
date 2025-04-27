package spark.kp4

import models.{GBTModelTrainer, RandomForestTrainer}
import preprocessor.DataPreprocessor

import org.apache.spark.sql.SparkSession


object Main {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("AntiFraud Training")
      .master("local[*]")
      .config("spark.executor.memory", "8g")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.executorEnv.PYSPARK_PYTHON", "/usr/local/bin/python3")
      .config("spark.pyspark.driver.python", "/usr/local/bin/python3")
      .getOrCreate()

    // Register shutdown hook for graceful termination
    Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run(): Unit = {
        println("Shutdown hook triggered, stopping Spark session...")
        if (!spark.sparkContext.isStopped) {
          spark.stop()
        }
      }
    })

    val preprocessor = new DataPreprocessor(spark)

    if (!preprocessor.isDataProcessed()) {
      println("processing data")
      preprocessor.preProcessData("train")
      preprocessor.preProcessData("test")
    }
    val trainer = new GBTModelTrainer(spark)
    trainer.trainAndEvaluate()
  }


}
