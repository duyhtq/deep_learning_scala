package robotbase.brain.function

import robotbase.brain.T

trait Function {

  def compute(x: Double): Double

  def differentiate(x: Double): Double

  def apply(x: T): T = {
    val cx = x.copy
    update(cx, compute, 0, 0)
    cx
  }

  def d(x: T): T = {
    val cx = x.copy
    update(cx, differentiate, 0, 0)
    cx
  }

  // travel a N-dimensional space
  def update(x: T, f: Double => Double, k: Int, index: Int): Unit = {
    if (k == x.dim.length) x.update(index, f(x(index)))
    else for (i <- 0 until x.dim(k))
      update(x, f, k + 1, index * x.dim(k) + i)
  }
}

object Function {
  def apply(name: String): Function = name.toLowerCase match {
    case "sigmoid" => Sigmoid
    case "relu" => ReLU
  }
}