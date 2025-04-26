package spark.kp4
package models

import preprocessor.FeaturePreprocessor
import utils.EvaluationUtils

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, SparkSession}

class RandomForestTrainer(spark: SparkSession) {

  val modelPath = "models/rf_model_with_weights"

  def trainAndEvaluate(): Unit = {
    val trainData = spark.read
      .option("header", "true")
      .schema(FeaturePreprocessor.schema.add("isFraud", "integer"))
      .option("inferSchema", "true")
      .csv("data/processed/train/processed_train_data.csv")

    val Array(trainSplit, testSplit) = trainData.randomSplit(Array(0.8, 0.2), seed = 1409)
    val assembledTrainData = addClassWeights(FeaturePreprocessor.assembleData(trainSplit))
    val assembledTestData = FeaturePreprocessor.assembleData(testSplit)

    loadModel(modelPath) match {
      case Some(existingModel) =>
        println("Loaded existing model.")
        val predictions = existingModel.transform(assembledTestData)
        EvaluationUtils.evaluate(predictions)
      case None =>
        println("No model found. Training a new one.")
        val model = trainModel(assembledTrainData)
        model.write.overwrite().save(modelPath)
        println("New model saved.")
        val predictions = model.transform(assembledTestData)
        EvaluationUtils.evaluate(predictions)
    }
  }

  private def addClassWeights(data: DataFrame): DataFrame = {
    // Calculate class weights
    val classCounts = data.groupBy("isFraud").count()
    val totalSamples = data.count()

    val classWeights = classCounts.withColumn(
      "classWeight",
      lit(totalSamples) / col("count")
    ).select(col("isFraud").alias("fraudClass"), col("classWeight"))

    // Join weights with the original dataset
    data.join(classWeights, data("isFraud") === classWeights("fraudClass"))
      .withColumn("weight", col("classWeight"))
      .drop("fraudClass", "classWeight")
  }

  private def loadModel(modelPath: String): Option[RandomForestClassificationModel] = {
    try {
      println(s"Trying to load model from $modelPath")
      Some(RandomForestClassificationModel.load(modelPath))
    } catch {
      case e: Exception =>
        println(s"Model loading failed: ${e.getMessage}")
        None
    }
  }

  private def trainModel(data: DataFrame): RandomForestClassificationModel = {
    val rfClassifier = new RandomForestClassifier()
      .setLabelCol("isFraud")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setWeightCol("weight") // Use the weight column
      .setNumTrees(100)
      .setMaxDepth(12)
      .setMaxBins(32)

    rfClassifier.fit(data)
  }
}
