package robotbase.brain.function

object ReLU extends Function {

  def compute(x: Double): Double = Math.max(0, x)

  def differentiate(x: Double): Double = if (x > 0) 1 else 0

}