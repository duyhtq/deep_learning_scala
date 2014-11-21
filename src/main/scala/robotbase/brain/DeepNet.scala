package robotbase.brain

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.SparkContext
import robotbase.brain.function.Function
import robotbase.brain.layer.{Full, Input, Regression}

class DeepNet(sc: SparkContext, configFile: String) {

  val c = ConfigFactory.load(configFile)
  C.update(c) // update to a global singleton configuration object

  U.log("building net")
  var lastDim = Array[Int]()
  var currentDim = Array[Int]()
  val net = c.getConfigList("net.layer").toArray.map(ci => {
    val layer = ci.asInstanceOf[Config]
    lastDim = currentDim.clone
    currentDim = U.fromJavaListToScalaArray(layer.getIntList("dim"))
    layer.getString("type").toLowerCase match {
      case "input" => new Input(currentDim)
      case "full" => new Full(Function(layer.getString("function")), lastDim(0), currentDim(0))
      case "regression" => new Regression
    }
  })

  U.log("load training data")
  val trainData = sc.textFile(c.getString("data.trainFile")).map { line =>
    val a = line.split(" ").map(_.toDouble)
    val y = new Array[Double](C.nClass)
    y.update(a(0).toInt, 1)
    val x = a.slice(1, a.length).map(z => z / C.normalizationFactor)
    (new T(x, 28, 28, 1), new T(y))
  }.collect()
  C.trainSize = trainData.length

  U.log("loading test data")
  val testData = sc.textFile(c.getString("data.testFile")).map { line =>
    val a = line.split(" ").map(_.toDouble)
    val y = new Array[Double](C.nClass)
    y.update(a(0).toInt, 1)
    val x = a.slice(1, a.length).map(z => z / C.normalizationFactor)
    (new T(x, 28, 28, 1), new T(y))
  }.take(C.testSize)

  def run {
    U.log("start training...")
    var count = 0

    val start = System.nanoTime()

    while (count < C.nIteration) {
      val train = trainData((Math.random() * trainData.length).toInt)

      net(0).out = train._1
      for (i <- 1 until net.size)
        net(i).forward(net(i - 1).out)

      var loss = train._2
      for (i <- net.size - 1 until 0 by -1)
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
            net(i).forward(net(i - 1).out)

          if (ty.values(net(net.size - 1).out.argmax) == 1)
            nHits += 1

          testLoss += (net(net.size - 1).out - ty).values.map(i => i * i).sum

        }
        U.log("hits: " + 100.0 * nHits / testData.length + "% loss: " + testLoss + " time: " + (System.nanoTime() - start) / 1000000000 + "s")
      }
    }
  }

}
