package robotbase.brain.function

object Sigmoid extends Function {

    def compute(x: Double): Double = 1 / (1 + Math.exp(-x))

    def differentiate(x: Double): Double = {
      val t = compute(x)
      t * (1 - t)
    }

}
