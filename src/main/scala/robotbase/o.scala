package robotbase

import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}

import scala.collection.mutable.ArrayBuffer

object o {

  trait function {

    def compute(x: Double): Double

    def differentiate(x: Double): Double

    def apply(x: Tensor): Tensor = {
      val cx = x.copy
      update(cx, compute, 0, 0)
      cx
    }

    def d(x: Tensor): Tensor = {
      val cx = x.copy
      update(cx, differentiate, 0, 0)
      cx
    }

    // travel a N-dimensional space
    def update(x: Tensor, f: (Double) => Double, k: Int, index: Int): Unit = {
      if (k == x.dim.length) x.update(index, f(x(index)))
      else for (i <- 0 until x.dim(k))
        update(x, f, k + 1, index * x.dim(k) + i)
    }
  }

  object sigmoid extends function {
    def compute(x: Double): Double = 1 / (1 + Math.exp(-x))
    def differentiate(x: Double): Double = {
      val t = compute(x)
      t * (1 - t)
    }
  }

  object relu extends function {
    def compute(x: Double): Double = Math.max(0, x)
    def differentiate(x: Double): Double = if (x > 0) 1 else 0
  }

  def argmax(x: Tensor): Int = x.a.indexOf(x.a.max)

  def +(x: Tensor, y: Tensor): Tensor = new Tensor((x.a, y.a).zipped.map(_ + _), x.dim.clone)

  def -(x: Tensor, y: Tensor): Tensor = new Tensor((x.a, y.a).zipped.map(_ - _), x.dim.clone)

  def *+(x: Tensor, y: Tensor): Tensor = new Tensor((x.a, y.a).zipped.map(_ * _), x.dim.clone)

  def *(d: Double, x: Tensor): Tensor = new Tensor(x.a.map(ai => ai * d), x.dim.clone)

  def *(x: Tensor, y: Tensor): Tensor = {
    // special case: vector x matrix
    val a = ArrayBuffer[Double]()
    var i = 0
    var j = 0
    var d = 0.0
    while (j < y.length) {
      d += x(i) * y(j)
      j += 1
      i += 1
      if (i == x.length) {
        a.append(d)
        i = 0
        d = 0
      }
    }
    new Tensor(a.toArray)
  }

  def x(x: Tensor, y: Tensor): Tensor = {
    // special case: vector x vector
    val r = new ArrayBuffer[Double]()
    var j = 0
    while (j < y.length) {
      var i = 0
      while (i < x.length) {
        r.append(x(i) * y(j))
        i += 1
      }
      j += 1
    }
    new Tensor(r.toArray, x.length, y.length)
  }

  def t(x: Tensor): Tensor = {
    // special case: matrix
    val r = new Tensor(x.dim(1), x.dim(0))
    var i = 0
    while (i < r.dim(0)) {
      var j = 0
      while (j < r.dim(1)) {
        val index = i + j * r.dim(0)
        r.update(index, x(index))
        j += 1
      }
      i += 1
    }
    r
  }























  def *(x: Vector, w: Matrix): Vector = {
    val a = new ArrayBuffer[Double]()
    var i = 0
    var j = 0
    var d = 0.0
    val wa = w.toArray
    while (j < wa.length) {
      d += x.apply(i) * wa.apply(j)
      i += 1
      j += 1
      if (i == x.size) {
        a.append(d)
        i = 0
        d = 0.0
      }
    }

    Vectors.dense(a.toArray)
  }

  def t(m: Matrix): Matrix = {
    val r = Matrices.dense(m.numCols, m.numRows, Vectors.zeros(m.numCols * m.numRows).toArray)
    var i = 0
    while (i < r.numRows) {
      var j = 0
      while (j < r.numCols) {
        r.toArray.update(i + j * r.numRows, m.toArray.apply(j + i * m.numRows))
        j += 1
      }
      i += 1
    }
    r
  }


  def x(a: Vector, b: Vector): Matrix = {
    val r = new ArrayBuffer[Double]()
    var j = 0
    while (j < b.size) {
      var i = 0
      while (i < a.size) {
        r.append(a.apply(i) * b.apply(j))
        i += 1
      }
      j += 1
    }
    Matrices.dense(a.size, b.size, r.toArray)
  }

  def *(a: Vector, b: Vector): Double = {
    var sum = 0.0
    var i = 0
    while (i < a.size) {
      sum += a.apply(i) * b.apply(i)
      i += 1
    }
    sum
  }

  def *(d: Double, x: Vector): Vector = {
    val a = new ArrayBuffer[Double]()
    var i = 0
    while (i < x.size) {
      a.append(x.apply(i) * d)
      i += 1
    }
    Vectors.dense(a.toArray)
  }

  def *(d: Double, x: Matrix): Matrix = {
    val a = new ArrayBuffer[Double]()
    val t = x.toArray
    var i = 0
    while (i < t.size) {
      a.append(t.apply(i) * d)
      i += 1
    }
    Matrices.dense(x.numRows, x.numCols, a.toArray)
  }

  def +(a: Vector, b: Vector): Vector = {
    var i = 0
    val t = new ArrayBuffer[Double]()
    while (i < a.size) {
      t.append(a.apply(i) + b.apply(i))
      i += 1
    }
    Vectors.dense(t.toArray)
  }

  def -(a: Vector, b: Vector): Vector = {
    var i = 0
    val t = new ArrayBuffer[Double]()
    while (i < a.size) {
      t.append(a.apply(i) - b.apply(i))
      i += 1
    }
    Vectors.dense(t.toArray)
  }

  def +(a: Matrix, b: Matrix): Matrix = {
    val aa = a.toArray
    val ba = b.toArray
    val r = new ArrayBuffer[Double]()
    for (i <- 0 until aa.length) {
      r.append(aa.apply(i) + ba.apply(i))
    }
    Matrices.dense(a.numRows, a.numCols, r.toArray)
  }

  def -(a: Matrix, b: Matrix): Matrix = {
    val aa = a.toArray
    val ba = b.toArray
    val r = new ArrayBuffer[Double]()
    var i = 0
    while (i < aa.length) {
      //println(a.numRows + " " + a.numCols + " " + b.numRows + " " + b.numCols)
      val aaa = aa.apply(i)
      val bbb = ba.apply(i)
      r.append(aaa - bbb)
      i += 1
    }
    Matrices.dense(a.numRows, a.numCols, r.toArray)
  }

  def *+(a: Vector, b: Vector): Vector = {
    val r = new ArrayBuffer[Double]()
    var i = 0
    while (i < a.size) {
      r.append(a.apply(i) * b.apply(i))
      i += 1
    }
    Vectors.dense(r.toArray)
  }

}
