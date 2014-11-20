package robotbase

case class Tensor(dim: Array[Int]) {

  var a = new Array[Double](dim.reduce(_ * _))

  def this(seed: String, dim: Array[Int]) {
    this(dim)
    if (seed == R.random)
      for (i <- 0 until a.length)
        a.update(i, (Math.random() * 1.0 / a.length) * {if (Math.random() < 0.5) -1 else 1})
  }

  def this(a: Array[Double], dim: Array[Int]) {
    this(dim)
    this.a = a
  }

  def this(a: Array[Double]) {
    this(a, Array(a.length))
  }

  def this(dim: Int*) = this(dim.toArray)

  def this(seed: String, dim: Int*) = this(seed, dim.toArray)

  def this(a: Array[Double], dim: Int*) = this(a, dim.toArray)

  def copy = new Tensor(a.clone, dim.clone)

  def copyDimensions = new Tensor(dim.clone)
  
  def reset = for (i <- 0 until a.length) a(i) = 0

  def apply(loc: Array[Int]): Double = a(index(loc))

  def apply(i: Int): Double = a(i)

  def update(loc: Array[Int], x: Double) = a.update(index(loc), x)

  def update(i: Int, x: Double) = a.update(i, x)

  def length = a.length

  def +(loc: Array[Int], x: Double) = {
    val i = index(loc)
    a.update(i, a(i) + x)
  }

  def index(loc: Array[Int]): Int = {
    var x = 0
    for (i <- 0 until dim.length)
      x = x * dim(i) + loc(i)
    x
  }

  def +(d: Tensor): Tensor = {
    null
  }
}