package com.github.vangj.spark.svm

import org.apache.log4j.LogManager
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * Stochastic gradient descent algorithm for learning a SVM hyper-plane that can linearly separate
  * postive (+1) and negative (-1) classes. Uses the
  * <a href="http://www.ee.oulu.fi/research/imag/courses/Vedaldi/ShalevSiSr07.pdf">Pegasos algorithm</a>.
  */
object Sgd {
  @transient lazy val logger = LogManager.getLogger(Sgd.getClass)

  /**
    * Parameters.
    * @param T Max number of iterations.
    * @param k Max samples per batch.
    * @param lambda Regularization parameter.
    * @param seed Seed for random number generator.
    * @param delim Delimiter for input text.
    * @param debug A boolean to see if debugging is turned on.
    * @param input Input path.
    * @param output Output path.
    */
  case class Params(
                     T: Int = 10,
                     k: Int = 1,
                     lambda: Double = 0.5d,
                     seed: Long = 37L,
                     delim: String = ",",
                     debug: Boolean = false,
                     input: String = "",
                     output: String = "")

  /**
    * A vector.
    * @param points List of doubles.
    */
  case class Vector(points: List[Double]) {
    def size(): Int = points.length

    def *(that: Vector): Double = {
      (this.points zip that.points).map(p => p._1 * p._2).sum
    }

    def +(that: Vector): Vector = {
      new Vector((this.points zip that.points).map(p => p._1 + p._2))
    }

    def *(scalar: Double): Vector = {
      new Vector(points.map(_ * scalar))
    }

    def l2norm(): Double = {
      Math.sqrt(points.map(Math.pow(_, 2.0d)).sum)
    }

    override def toString: String = s"[${points.map(d => "%.5f".format(d)).mkString(",")}]"
  }

  /**
    * A data point with (x, y) where x is a vector and y is a scalar.
    * @param y Scalar representing the class. e.g. 1 or -1.
    * @param x A vector.
    * @param result Place-holder for multiplication result. A convenience field that will probably
    *               confuse the user.
    */
  case class DataPoint(y: Double, x: Vector, result: Double = 0.0d) {
    def *(w: Vector): DataPoint = {
      val r = y * (w * x)
      new DataPoint(y, x, r)
    }

    override def toString: String = s"${y} ${x.toString}"
  }


  /**
    * Starts the stochastic gradient descent.
    * @param data RDD of data points.
    * @param params Parameters.
    * @return Estimated weights for SVM.
    */
  def learnHyperplane(data: RDD[DataPoint], params: Params): (Double, Vector) = {
    val lambda = params.lambda
    val seed = params.seed

    val N = data.count().toDouble
    val k = Math.min(N, params.k)
    val pct = k / N

    var b = 0.0d
    var w = getInitialWeights(data.first().x.size, lambda)

    for (t <- 1 to params.T) {
      val A = data.sample(false, pct, seed)
      val A_p = A.map(_ * w).filter(_.result < 1.0d)

      if (A_p.count() > 0) {
        val w_next = step(A_p, k.toInt, t, lambda, w)

        if (params.debug || t == params.T) {
          val b_next = (-1.0d / k.toDouble) * A_p.map(_.y).sum

          if (params.debug) {
            val iter = "%03d".format(t)
            val w_norm_next = w_next.l2norm()
            val w_norm = "%.5f".format(w_norm_next)
            val b_formatted = "%.5f".format(b_next)
            logger.debug(s"iter ${iter} w is ${w_next} at ${w_norm} with b ${b_formatted}")
          }

          b = b_next
        }

        w = w_next
      }

    }

    (b, w)
  }

  /**
    * Takes a step to estimate the updated weights.
    * @param A RDD of sampled data points.
    * @param k The number of data points. e.g. |A|
    * @param t The t-th iteration.
    * @param lambda Regularization parameter.
    * @param w The current weights.
    * @return The updated weights.
    */
  private def step(A: RDD[DataPoint], k: Int, t: Int, lambda: Double, w: Vector): Vector = {
    val n = 1.0d / lambda / t.toDouble
    val w_t_half = (w * (1.0d - n * lambda)) + A.map(p => p.x * (p.y * n / k.toDouble)).reduce(_ + _)
    w_t_half * Math.min(1.0d, (1.0d / Math.sqrt(lambda)) / w_t_half.l2norm())
  }

  /**
    * Gets the initial weights. All zeros for now. The original algorithm specifies that
    * ||w|| is less than or equal to 1 / sqrt(lambda). Setting them all to zero will certainly
    * qualifies.
    * @param dims The number of columns (dimensions). Should NOT include the class variable.
    * @param lambda Lambda (regularization parameter).
    * @return List of initial weights.
    */
  private def getInitialWeights(dims: Int, lambda: Double): Vector = {
    new Vector(List.fill(dims)(0.0d))
  }

  /**
    * Converts an RDD of strings to an RDD of list of data points.
    * @param data RDD.
    * @param delim Delimiter. e.g. tab, comma, space, etc...
    * @return RDD.
    */
  private def convert(data: RDD[String], delim: String): RDD[DataPoint] = {
    data.map(s => {
      val dataPoints = s.split(delim).map(_.toDouble)
      new DataPoint(dataPoints(0), new Vector(dataPoints.tail.toList))
    })
  }

  def main(args: Array[String]): Unit = {
    val parser = new scopt.OptionParser[Params]("Sgd") {
      head("Sgd", "0.0.1")
      opt[Int]("T").required().action( (x, c) => c.copy(T = x)).text("number of iterations")
      opt[Int]("k").required().action( (x, c) => c.copy(k = x)).text("number of samples")
      opt[Double]("lambda").required().action( (x, c) => c.copy(lambda = x)).text("regularization parameter (learning rate)")
      opt[Long]("seed").required().action( (x, c) => c.copy(seed = x)).text("seed for randomization")
      opt[String]("delim").required().action( (x, c) => c.copy(delim = x)).text("delimiter for input file")
      opt[Boolean]("debug").required().action( (x, c) => c.copy(debug = x)).text("flag to turn on debugging output")
      opt[String]("input").required().action( (x, c) => c.copy(input = x)).text("input path")
      opt[String]("output").required().action( (x, c) => c.copy(output = x)).text("output path")
    }

    parser.parse(args, Params()) match {
      case Some(params) =>
        val conf = new SparkConf().setAppName(s"sgd for svm from ${params.input} to ${params.output}")
        val spark = SparkSession.builder().config(conf).getOrCreate()
        val data = convert(spark.sparkContext.textFile(params.input), params.delim)

        val results = learnHyperplane(data, params)
        val b = results._1
        val w = results._2

        spark.sparkContext.parallelize(List(s"${b} ${w.toString}")).saveAsTextFile(params.output)

        if (params.debug) {
          data.foreach(p => {
            val r = p.x * w - b
            val b_formatted = "%.5f".format(b)
            val r_formatted = "%.5f".format(r)
            val isCorrect = if ((p.y < 0 && r < 0) || (p.y > 0 && r > 0)) true else false
            logger.debug(s"${p.x} * ${w} - ${b_formatted} = ${r_formatted}, class = ${p.y}, correct ${isCorrect}")
          })
        }

        spark.stop
      case None =>
        logger.error("invalid arguments!")
    }
  }
}
