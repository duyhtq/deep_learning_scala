package rb.ai

import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

object Test extends App {
  val v = Vectors.dense(1, 2, 3)
  println(v)
  v.toArray.update(0, 4)
  println(v)
  val m = Matrices.dense(3, 2, Array(0, 1, 2, 3, 4, 5))
  println(m)
  val r = f.t(m)
  println(r)
  val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("test"))
  val a = sc.parallelize(List(1, 2, 3, 4, 5))
  a.takeSample(true, 2).foreach(println)
  a.takeSample(true, 2).foreach(println)

  println("---")
  for (i <- 2 until 9)
    println(i)
}
