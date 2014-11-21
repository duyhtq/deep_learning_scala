package robotbase.brain.layer

import robotbase.brain.function.Function
import robotbase.brain.{R, T}

class Convolution(f: Function, k: Int, n: Int, d: Int, d0: Int) extends Layer(f) {

  x = new T(n, n, d)
  b = new T(R.random, k, k, d, d0)
  w = new T(R.random, k, k, d, d0)
  db = new T(k, k, d, d0)
  dw = new T(k, k, d, d0)

  def forward(in: T) {
    this.in = in
    for (h <- 0 until d)
      for (i <- 0 until n)
        for (j <- 0 until n) {
          var sum = 0.0
          for (t <- 0 until d0)
            for (u <- 0 until k)
              for (v <- 0 until k) {
                sum += in(Array(i + u, j + v, t)) * w(Array(u, v, h, t)) + b(Array(u, v, h, t))
              }
          x.update(Array(i, j, h), sum)
        }
    out = f(x)
  }

  def backward(loss: T): T = {
    val gradient = f.d(x) *+ loss
    val newLoss = in.copyDim
    for (h <- 0 until d)
      for (i <- 0 until n)
        for (j <- 0 until n) {
          val g = gradient(i, j, h)
          for (t <- 0 until d0)
            for (u <- 0 until k)
              for (v <- 0 until k) {
                dw.+(Array(u, v, h, t), in(Array(i + u, j + v, t)) * g)
                db.+(Array(u, v, h, t), g)
                newLoss.+(Array(i + u, j + v, t), g * w(Array(u, v, h, t)))
              }
        }
    newLoss
  }

}
