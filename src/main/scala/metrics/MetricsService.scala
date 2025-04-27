package spark.kp4.metrics

import io.prometheus.client.{Counter, Gauge, Histogram, Summary}
import io.prometheus.client.exporter.HTTPServer
import io.prometheus.client.hotspot.DefaultExports
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import java.io.IOException

/**
 * Service for collecting and exposing Prometheus metrics
 */
object MetricsService {
  // Initialize HTTP server to expose metrics
  private var server: Option[HTTPServer] = None

  def generateBatchMetrics(batchDF: DataFrame): Unit = {
    val batchCount = batchDF.count()

    val batchTimer = MetricsService.batchProcessingTime.startTimer()

    MetricsService.batchSize.set(batchCount)

    MetricsService.activeTransactions.inc(batchCount)

    try {

      MetricsService.transactionsProcessed.inc(batchCount)

      val fraudCount = batchDF.filter(col("isFraud") === true).count()
      MetricsService.fraudTransactionsDetected.inc(fraudCount)

      val highRiskCount = batchDF.filter(col("isHighRisk") === true).count()
      MetricsService.highRiskTransactionsDetected.inc(highRiskCount)

      if (batchCount > 0) {
        val approvalRate = (batchCount - fraudCount).toDouble / batchCount
        MetricsService.transactionApprovalRate.set(approvalRate)
      }

    } finally {
      MetricsService.activeTransactions.dec(batchCount)

      batchTimer.observeDuration()
    }
  }

  // Transaction metrics
  val transactionsProcessed: Counter = Counter.build()
    .name("transactions_processed_total")
    .help("Total number of transactions processed")
    .register()

  val fraudTransactionsDetected: Counter = Counter.build()
    .name("fraud_transactions_detected_total")
    .help("Total number of fraud transactions detected")
    .register()

  val highRiskTransactionsDetected: Counter = Counter.build()
    .name("high_risk_transactions_detected_total")
    .help("Total number of high-risk transactions detected (probability > 0.4)")
    .register()

  val batchProcessingTime: Histogram = Histogram.build()
    .name("batch_processing_time_seconds")
    .help("Time taken to process a batch of transactions")
    .buckets(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
    .register()

  val batchSize: Gauge = Gauge.build()
    .name("batch_size")
    .help("Number of transactions in the current batch")
    .register()

  val activeTransactions: Gauge = Gauge.build()
    .name("active_transactions")
    .help("Number of transactions currently being processed")
    .register()

  val transactionApprovalRate: Gauge = Gauge.build()
    .name("transaction_approval_rate")
    .help("Rate of approved transactions")
    .register()

  /**
   * Start the metrics server
   * @param port Port to expose metrics on
   */
  def startServer(port: Int = 9091): Unit = {
    if (server.isEmpty) {
      try {
        // Register JVM metrics
        DefaultExports.initialize()

        // Start HTTP server
        server = Some(new HTTPServer(port))
        println(s"Metrics server started on port $port")
      } catch {
        case e: IOException =>
          println(s"Failed to start metrics server: ${e.getMessage}")
      }
    }
  }

  /**
   * Stop the metrics server
   */
  def stopServer(): Unit = {
    server.foreach { s =>
      s.stop()
      println("Metrics server stopped")
    }
    server = None
  }
}
