import sbt.Keys.excludeDependencies

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

val sparkVersion = "3.5.0"
val prometheusVersion = "0.16.0"

lazy val root = (project in file("."))
  .settings(
    name := "bigdataKp4",
    idePackagePrefix := Some("spark.kp4"),
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-streaming" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion,

      // Prometheus client for metrics collection
      "io.prometheus" % "simpleclient" % prometheusVersion,
      "io.prometheus" % "simpleclient_hotspot" % prometheusVersion,
      "io.prometheus" % "simpleclient_httpserver" % prometheusVersion,
      "io.prometheus" % "simpleclient_pushgateway" % prometheusVersion,

      // Elasticsearch Spark Connector
      "org.elasticsearch" % "elasticsearch-spark-30_2.12" % "7.17.10"
    )

  )
