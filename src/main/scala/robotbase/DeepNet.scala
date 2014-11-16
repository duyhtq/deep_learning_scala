package rb.ai

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

object DeepNet extends App {

  val start = System.nanoTime()
  Logger.getLogger("org").setLevel(Level.WARN)
  val sc = new SparkContext(new SparkConf().setMaster("local[8]").setAppName("deep net").set("spark.executor.memory", "12g"))

  /*
      val C = 2
      val R = 1
      var N = 100000000
      val S: Double = 3.0
      val K = 100
      val B = 2
      val file = "/media/ll/4e816b11-0b58-48b3-a8d8-c460a2971b3f/downloads/a.txt"
      var net = Array(robotbase.deepnet.Layer(0, 2), robotbase.deepnet.Layer(2, 3), robotbase.deepnet.Layer(3, 2))
  */


  val CLASS = 10
  val R = 255
  var ITERATIONS = 100000000
  val LEARNING_RATE: Double = 3.0
  val REPORT = 5000
  val BATCH_SIZE = 10
  val filename = "data/mnist/train50000.txt"
  var net = Array(Layer(0, 784), Layer(784, 60), Layer(60, 30), Layer(30, 10))

  println("loading data...")
  val train = sc.textFile(filename).map { line =>
    val a = line.split(" ").map(_.toDouble)
    val y = Vectors.zeros(CLASS)
    y.toArray.update(a(0).toInt, 1)
    val x = Vectors.dense(a.slice(1, a.length).map(z => z / R))
    (y, x)
  }
  val train_data = train.collect()

  println("start training...")
  var stats = 0
  var count = 0
  var loss = 0.0
  var b = 0

  while (ITERATIONS > 0) {
    ITERATIONS -= 1
    b += 1
    count += 1

    val data = train_data((Math.random() * train_data.size).toInt)
    val x = data._2
    val y = data._1

    net(0).a = x

    // forward prop
    var i = 1
    while (i < net.size) {
      val l = net(i)
      l.x = f.+(f.*(net(i - 1).a, l.w), l.b)
      l.a = f.sigmoid.forward(l.x)
      i += 1
    }

    // last layer (output)
    val last = net(net.size - 1)
    /*
    val e = last.x.toArray.map(ei => Math.exp(ei))
    val sum = e.reduce(_ + _)
    last.a = Vectors.dense(e.map(ei => ei / sum))
    var delta = f.-(last.a, y)
*/
    var delta = f.**(f.sigmoid.backward(last.x), f.-(last.a, y))
    last.db = f.+(last.db, delta)
    last.dw = f.+(last.dw, f.x(net(net.size - 2).a, delta))


    // back prop
    i = net.size - 2
    while (i > 0) {
      val layer = net(i)
      val x_prime = f.sigmoid.backward(layer.x)
      delta = f.**(x_prime, f.*(delta, f.t(net(i + 1).w)))
      layer.db = f.+(layer.db, delta)
      layer.dw = f.+(layer.dw, f.x(net(i - 1).a, delta))
      i -= 1
    }


    // calculate loss
    var k = 0
    while (k < CLASS) {
      loss -= y(k) * Math.log(last.a(k))
      k += 1
    }

    // stats
    if (y(last.a.toArray.indexOf(last.a.toArray.max)) == 1) stats += 1

    if (count == REPORT) {
      println(100 * stats / REPORT + "% " + loss + " " + (System.nanoTime() - start) / 1000000000.0 + " seconds")
      count = 0
      stats = 0
      loss = 0
    }

    if (b % BATCH_SIZE == 0) {

      // update
      var i = 0
      while (i < net.size) {
        val l = net(i)
        l.w = f.-(l.w, f.*(LEARNING_RATE / BATCH_SIZE, l.dw))
        l.b = f.-(l.b, f.*(LEARNING_RATE / BATCH_SIZE, l.db))
        l.db = Vectors.zeros(l.db.size)
        l.dw = Matrices.dense(l.dw.numRows, l.dw.numCols, Vectors.zeros(l.dw.numRows * l.dw.numCols).toArray)
        i += 1
      }

    }
  }

  sc.stop()
  println("total time: " + (System.nanoTime() - start) / 1000000000.0 + " seconds")
}
