package robotbase

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object DeepNet extends App {

  val start = System.nanoTime()
  Logger.getLogger("org").setLevel(Level.WARN)
  val sc = new SparkContext(new SparkConf().setMaster("local[8]").setAppName("deep net").set("spark.executor.memory", "12g"))


  val net = Array(
    new Full(o.sigmoid, 0, 784),
    new Full(o.sigmoid, 784, 100),
    new Full(o.sigmoid, 100, 10),
    new Regression)


/*
  val net = Array(
    new Full(o.relu, 0, 784),
    new Convolution(o.relu, 5, 24, 8, 1),
    new Pooling(2, 12, 8),
    new Convolution(o.relu, 5, 8, 16, 8),
    new Pooling(2, 4, 16),
    new Full(o.relu, 4 * 4 * 16, 10),
    new Softmax
  )
*/

  val trainFile = "data/mnist/train50000.txt"
  println("loading data...")
  val trainData = sc.textFile(trainFile).map { line =>
    val a = line.split(" ").map(_.toDouble)
    val y = new Array[Double](C.nClass)
    y.update(a(0).toInt, 1)
    val x = a.slice(1, a.length).map(z => z / C.normalizationFactor)
    (new Tensor(x, 28, 28, 1), new Tensor(y))
  }.collect()
  C.trainSize = trainData.length
  
  val testFile = "data/mnist/test10000.txt"
  println("loading data...")
  val testData = sc.textFile(testFile).map { line =>
    val a = line.split(" ").map(_.toDouble)
    val y = new Array[Double](C.nClass)
    y.update(a(0).toInt, 1)
    val x = a.slice(1, a.length).map(z => z / C.normalizationFactor)
    (new Tensor(x, 28, 28, 1), new Tensor(y))
  }.take(C.testSize)

  println("start training...")
  var count = 0

  while (count < C.nIteration) {
    val train = trainData((Math.random() * C.trainSize).toInt)

    net(0).out = train._1
    for (i <- 1 until net.size)
      net(i).forward(net(i-1).out)

    var loss = train._2
    for (i <- net.size-1 until 0 by -1)
      loss = net(i).backward(loss)

    count += 1

    if (count % C.batchSize == 0)
      for (i <- 1 until net.size - 1)
        net(i).update // update weight * bias

    if (count % C.reportSize == 0) {
      var nHits = 0
      var testLoss = 0.0
      for (test <- testData) {
        val ty = test._2

        net(0).out = test._1
        for (i <- 1 until net.size)
          net(i).forward(net(i-1).out)

        if (ty.a(o.argmax(net(net.size - 1).out)) == 1)
          nHits += 1

        testLoss += o.-(net(net.size-1).out, ty).a.map(i => i * i).sum

      }
      println("hits: " + 100.0 * nHits / C.testSize + "% loss: " + testLoss + " time: " + (System.nanoTime() - start) / 1000000000 + "s")
    }
  }

  sc.stop()
  println("total time: " + (System.nanoTime() - start) / 1000000000.0 + " seconds")
}
