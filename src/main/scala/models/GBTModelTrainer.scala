package spark.kp4
package models

import preprocessor.FeaturePreprocessor
import utils.EvaluationUtils

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.sql.{DataFrame, SparkSession}

object GBTModelTrainer {
  val modelPath = "models/gbt_model_with_weights"
}

/*
Accuracy: 0.9789479916434776
ROC: 0.9176427284649091
AUPR: 0.6501516957662576
F1: 0.9756496357633486
 */
class GBTModelTrainer(spark: SparkSession) {

  def trainAndEvaluate(): Unit = {
    val trainData = spark.read
      .option("header", "true")
      .schema(FeaturePreprocessor.schema.add("isFraud", "integer"))
      .option("inferSchema", "true")
      .csv("data/processed/train/processed_train_data.csv")

    val Array(trainSplit, testSplit) = trainData.randomSplit(Array(0.7, 0.3), seed = 1409)
    val assembledTrainData = FeaturePreprocessor.assembleData(trainSplit)
    val assembledTestData = FeaturePreprocessor.assembleData(testSplit)

    loadModel(GBTModelTrainer.modelPath) match {
      case Some(existingModel) =>
        println("Loaded existing model.")
        val predictions = existingModel.transform(assembledTestData)
        EvaluationUtils.evaluate(predictions)
      case None =>
        println("No model found. Training a new one.")
        val model = trainModel(assembledTrainData)
        model.write.overwrite().save(GBTModelTrainer.modelPath)
        println("New model saved.")
        val predictions = model.transform(assembledTestData)
        EvaluationUtils.evaluate(predictions)
    }
  }

  private def loadModel(modelPath: String): Option[GBTClassificationModel] = {
    try {
      println(s"Trying to load model from $modelPath")
      Some(GBTClassificationModel.load(modelPath))
    } catch {
      case e: Exception =>
        println(s"Model loading failed: ${e.getMessage}")
        None
    }
  }

  private def trainModel(data: DataFrame): GBTClassificationModel = {
    val gbtClassifier = new GBTClassifier()
      .setLabelCol("isFraud")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxDepth(12)
      .setStepSize(0.1)
      .setMaxBins(32)

    gbtClassifier.fit(data)
  }
}


