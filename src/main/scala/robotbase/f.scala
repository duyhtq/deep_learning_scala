package rb.ai

import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}

import scala.collection.mutable.ArrayBuffer

object f {

  object sigmoid {
    def forward(x: Double): Double = 1 / (1 + Math.exp(-x))
    def forward(v: Vector): Vector = Vectors.dense(v.toArray.map(x => forward(x)))
    def backward(x: Double): Double = forward(x) * (1 - forward(x))
    def backward(v: Vector): Vector = Vectors.dense(v.toArray.map(fx => backward(fx)))
    def backward(m: Matrix): Matrix = Matrices.dense(m.numRows, m.numCols, m.toArray.map(x => backward(x)))
  }

  object relu {
    def forward(x: Double): Double = Math.max(0, x)
    def forward(v: Vector): Vector = Vectors.dense(v.toArray.map(x => forward(x)))
    def backward(x: Double): Double = if (x == 0) 0 else 1
    def backward(v: Vector): Vector = Vectors.dense(v.toArray.map(fx => backward(fx)))
  }


  def randomVector(n: Int): Vector = {
    val a = ArrayBuffer[Double]()
    var i = 0
    while (i < n) {
      a.append((Math.random() - 0.5) * 2)
      i += 1
    }
    Vectors.dense(a.toArray)
  }

  def randomMatrix(n: Int, m: Int): Matrix = {
    val a = ArrayBuffer[Double]()
    var i = n * m
    while (i > 0) {
      a.append((Math.random() - 0.5) * 2)
      i -=1
    }
    Matrices.dense(n, m, a.toArray)
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

  def **(a: Vector, b: Vector): Vector = {
    val r = new ArrayBuffer[Double]()
    var i = 0
    while (i < a.size) {
      r.append(a.apply(i) * b.apply(i))
      i += 1
    }
    Vectors.dense(r.toArray)
  }

}
