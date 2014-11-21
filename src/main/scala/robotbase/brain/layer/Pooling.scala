package robotbase.brain.layer

import robotbase.brain.T

import scala.collection.mutable.ArrayBuffer

class Pooling(k: Int, n: Int, d: Int) extends Layer() {

  out = new T(n, n, d)
  var save = new ArrayBuffer[Array[Int]]()

  def forward(in: T) {
    this.in = in
    this.save = new ArrayBuffer[Array[Int]]()
    for (h <- 0 until d)
      for (i <- 0 until n)
        for (j <- 0 until n) {
          var max = Double.MinValue
          var maxi = -1
          var maxj = -1
          for (u <- 0 until k)
            for (v <- 0 until k) {
              val q = in(Array(i * k + u, j * k + v, h))
              if (q > max) {
                max = q
                maxi = i * k + u
                maxj = j * k + v
              }
            }
          save.append(Array(maxi, maxj, h))
          out.update(Array(i, j, h), max)
        }
  }

  def backward(loss: T): T = {
    val gradient = loss
    val newLoss = in.copyDim
    var u = 0
    for (h <- 0 until d)
      for (i <- 0 until n)
        for (j <- 0 until n) {
          newLoss.+(save(u), gradient(Array(i, j, h)))
          u += 1
        }
    newLoss
  }

  override def update = {
    // no update needed in pooling layer
  }
}