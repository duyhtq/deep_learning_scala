package robotbase

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.log4j.{Logger, Level}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
import robotbase.brain.LinearAlgebra

object Test extends App {
  Logger.getLogger("org").setLevel(Level.WARN)
  val v = Vectors.dense(1, 2, 3)
  println(v)
  v.toArray.update(0, 4)
  println(v)
  val m = Matrices.dense(3, 2, Array(0, 1, 2, 3, 4, 5))
  println(m)
  val r = LinearAlgebra.t(m)
  println(r)
  val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("test"))
  val a = sc.parallelize(List(1, 2, 3, 4, 5))
  a.takeSample(true, 2).foreach(println)
  a.takeSample(true, 2).foreach(println)

  val config = ConfigFactory.load("mlp")
  config.getConfigList("net.layers").toArray.foreach(a => {
    val b = a.asInstanceOf[Config]
    println(b.getString("function"))
  })
}
