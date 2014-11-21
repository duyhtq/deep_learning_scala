package robotbase.brain

import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkConf, SparkContext}

object MLP4MNIST extends App {
  Logger.getLogger("org").setLevel(Level.WARN)
  val sc = new SparkContext(new SparkConf().setMaster("local[8]").setAppName("deep net").set("spark.executor.memory", "12g"))

  new DeepNet(sc, "mlp").run

  sc.stop
}
