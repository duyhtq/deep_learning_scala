package robotbase.brain

import scala.collection.mutable.ArrayBuffer

case class T(dim: Array[Int]) {

  var values = new Array[Double](dim.reduce(_ * _))

  def this(seed: String, dim: Array[Int]) {
    this(dim)
    if (seed == R.random)
      for (i <- 0 until values.length)
        values.update(i, (Math.random() * 1.0 / values.length) * {if (Math.random() < 0.5) -1 else 1})
  }

  def this(a: Array[Double], dim: Array[Int]) {
    this(dim)
    this.values = a
  }

  def this(a: Array[Double]) {
    this(a, Array(a.length))
  }

  def this(dim: Int*) = this(dim.toArray)

  def this(seed: String, dim: Int*) = this(seed, dim.toArray)

  def this(a: Array[Double], dim: Int*) = this(a, dim.toArray)

  def copy = new T(values.clone, dim.clone)

  def copyDim = new T(dim.clone)
  
  def updateDim(newDim: Array[Int]) {
    for (i <- 0 until dim.length)
      dim.update(i, newDim(i))
  }
  
  def reset = for (i <- 0 until values.length) values(i) = 0

  def apply(loc: Array[Int]): Double = values(index(loc))

  def apply(i: Int*): Double = values(index(i.toArray))

  def apply(i: Int): Double = values(i)

  def update(loc: Array[Int], x: Double) = values.update(index(loc), x)

  def update(i: Int, x: Double) = values.update(i, x)

  def length = values.length

  def +(loc: Array[Int], x: Double) {
    val i = index(loc)
    values.update(i, values(i) + x)
  }

  def index(loc: Array[Int]): Int = {
    var x = 0
    for (i <- 0 until dim.length)
      x = x * dim(i) + loc(i)
    x
  }

  def argmax(): Int = values.indexOf(values.max)

  def +(y: T): T = new T((values, y.values).zipped.map(_ + _), dim.clone)

  def -(y: T): T = new T((values, y.values).zipped.map(_ - _), dim.clone)

  def *+(y: T): T = new T((values, y.values).zipped.map(_ * _), dim.clone)

  def *(d: Double): T = new T(values.map(ai => ai * d), dim.clone)

  def *(y: T): T = {
    // special case: vector x matrix
    val a = ArrayBuffer[Double]()
    var i = 0
    var j = 0
    var d = 0.0
    while (j < y.length) {
      d += values(i) * y(j)
      j += 1
      i += 1
      if (i == values.length) {
        a.append(d)
        i = 0
        d = 0
      }
    }
    new T(a.toArray)
  }

  def x(y: T): T = {
    // special case: vector x vector
    val r = new ArrayBuffer[Double]()
    var j = 0
    while (j < y.length) {
      var i = 0
      while (i < values.length) {
        r.append(values(i) * y(j))
        i += 1
      }
      j += 1
    }
    new T(r.toArray, length, y.length)
  }

  def tranpose: T = {
    // special case: matrix
    val r = new T(dim(1), dim(0))
    var i = 0
    while (i < r.dim(0)) {
      var j = 0
      while (j < r.dim(1)) {
        val index = i + j * r.dim(0)
        r.update(index, values(index))
        j += 1
      }
      i += 1
    }
    r
  }


}