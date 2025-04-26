package spark.kp4.utils

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame

object EvaluationUtils {
  def evaluate(predictions: DataFrame): Unit = {
    val binaryEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("isFraud")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderPR")

    val multiclassEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("isFraud")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = multiclassEvaluator.evaluate(predictions)
    multiclassEvaluator.setMetricName("f1")
    val f1 = multiclassEvaluator.evaluate(predictions)
    val aupr = binaryEvaluator.evaluate(predictions)
    binaryEvaluator.setMetricName("areaUnderROC")
    val roc = binaryEvaluator.evaluate(predictions)

    println(s"Accuracy: $accuracy")
    println(s"ROC: $roc")
    println(s"AUPR: $aupr")
    println(s"F1: $f1")

  }
}
