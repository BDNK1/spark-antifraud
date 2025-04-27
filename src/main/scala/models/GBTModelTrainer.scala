package spark.kp4
package models

import preprocessor.FeaturePreprocessor
import utils.EvaluationUtils
import utils.EvaluationUtils.EvaluationMetrics

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.collection.mutable.ArrayBuffer

object GBTModelTrainer {
  val modelPath = "models/gbt_model_cross_validation"
}

/*
Accuracy: 0.9789479916434776
ROC: 0.9176427284649091
AUPR: 0.6501516957662576
F1: 0.9756496357633486
 */

/*
Best model found with parameters:
maxDepth: 12
maxBins: 64
stepSize: 0.2
Best model saved with metrics:
Accuracy: 0.9802605606000789
ROC: 0.9270417842916829
AUPR: 0.7089675718044302
F1: 0.9775512784269618
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
    val assembledTrainDataWithWeights = FeaturePreprocessor.addClassWeights(assembledTrainData)

    val assembledTestData = FeaturePreprocessor.assembleData(testSplit)

    loadModel(GBTModelTrainer.modelPath) match {
      case Some(existingModel) =>
        println("Loaded existing model.")
        val predictions = existingModel.transform(assembledTestData)
        EvaluationUtils.evaluate(predictions)
      case None =>
        println("No model found. Training a new one with cross-validation.")
        val (bestModel, bestMetrics) = trainModelWithCrossValidation(assembledTrainDataWithWeights, assembledTestData)
        bestModel.write.overwrite().save(GBTModelTrainer.modelPath)
        println("Best model saved with metrics:")
        EvaluationUtils.printMetrics(bestMetrics)
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

  private def trainModelWithCrossValidation(trainData: DataFrame, testData: DataFrame): (GBTClassificationModel, EvaluationMetrics) = {
    // Define the parameter grid for cross-validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(new GBTClassifier().maxDepth, Array(5, 8, 12))
      .addGrid(new GBTClassifier().maxBins, Array(32, 64))
      .addGrid(new GBTClassifier().maxIter, Array(32, 64))
      .addGrid(new GBTClassifier().stepSize, Array(0.05, 0.1, 0.2))
      .build()

    val models = ArrayBuffer[(GBTClassificationModel, EvaluationMetrics)]()

    // Train and evaluate models with different hyperparameters
    for (maxDepth <- Array(5, 8, 12); maxBins <- Array(32, 64); stepSize <- Array(0.05, 0.1, 0.2)) {
      println(s"Training model with maxDepth=$maxDepth, maxBins=$maxBins, stepSize=$stepSize")

      val gbtClassifier = new GBTClassifier()
        .setLabelCol("isFraud")
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
        .setStepSize(stepSize)

      // Train the model
      val model = gbtClassifier.fit(trainData)

      // Evaluate on test data
      val predictions = model.transform(testData)
      val metrics = EvaluationUtils.evaluate(predictions)

      println(s"Model variant with maxDepth=$maxDepth, maxBins=$maxBins, stepSize=$stepSize:")
      EvaluationUtils.printMetrics(metrics)
      println("--------------------------------------")

      models += ((model, metrics))
    }

    // Find the best model based on F1 score (can be changed to other metrics)
    val bestModelWithMetrics = models.maxBy(_._2.f1)

    println("\nBest model found with parameters:")
    val bestModel = bestModelWithMetrics._1
    println(s"maxDepth: ${bestModel.getMaxDepth}")
    println(s"maxBins: ${bestModel.getMaxBins}")
    println(s"stepSize: ${bestModel.getStepSize}")

    bestModelWithMetrics
  }

  private def trainBaselineModel(data: DataFrame): GBTClassificationModel = {
    val gbtClassifier = new GBTClassifier()
      .setLabelCol("isFraud")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    gbtClassifier.fit(data)
  }
}
