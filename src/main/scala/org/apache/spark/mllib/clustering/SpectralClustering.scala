package org.apache.spark.mllib.clustering

import com.github.fommil.netlib.ARPACK
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import breeze.linalg.{DenseVector => BDV}
import org.netlib.util.{intW, doubleW}

/*
* an implementation of spectral clustring on spark
*
* */
case class BipartiteResult(userCluster:RDD[(Int,Int)],itemCluster:RDD[(Int,Int)])

/**
 * @param mode RatioCut or NCut
 * @param numberCluster number of clusters
 * */
class GraphCluster(var mode:String, var numberCluster:Int) extends Serializable{

  def this()=this("RatioCut",10)
  def setMode(m:String):this.type ={
    //    require(m.equals("RatioCut")||m.equals("NCut"))
    this.mode = m
    this
  }

  def setClusterNum(n:Int):this.type ={
    this.numberCluster = n
    this
  }

  var numberOfEigen = numberCluster

  /**
   * @param w similarity matrix
   * */
  private def calcLaplacianMatrix(w:RDD[(Int, Int, Double)]):RDD[(Int,Int,Double)]={
    mode match{
      case "RatioCut" => {
        w.map(x=>(x._1,(x._2,x._3))).groupByKey().
          flatMap(x=>({
          val sum = x._2.map(_._2).sum
          x._2.map(t=>(x._1,t._1,{
            if(t._1 == x._1) sum-t._2 else -t._2
          }))
        })).
          map(x=>((x._1,x._2),x._3)).
          groupByKey().
          map(x=>(x._1._1,x._1._2,x._2.sum))
      }
      case "NCut" => {
        val nCols = w.map(_._2).max()+1
        val diag = w.sparkContext.parallelize((0 until nCols).map(i=>(i,i,0.0)))

        w.union(diag).
          map(x=>((x._1,x._2),x._3)).
          groupByKey().
          map(x=>(x._1._1,x._1._2,x._2.sum)).
          map(x=>(x._1,(x._2,x._3))).groupByKey().
          flatMap(x=>({
          val sum = x._2.map(_._2).sum
          x._2.map(t=>(x._1,t._1,{
            if(t._1 == x._1) 1.0-t._2/sum else -t._2/sum
          }))
        })).
          map(x=>((x._1,x._2),x._3)).
          groupByKey().
          map(x=>(x._1._1,x._1._2,x._2.sum))
      }
      case "SymNCut" => {
        val d = w.map(x=>(x._1,(x._2,x._3))).
          groupByKey().
          map(x=>(x._1,x._2.map(_._2).sum)).
          sortBy(_._1).
          collect().
          map(_._2)

        val nCols = w.map(_._2).max()+1
        val diag = w.sparkContext.parallelize((0 until nCols).map(i=>(i,i,0.0)))

        w.union(diag).
          map(x=>((x._1,x._2),x._3)).
          groupByKey().
          map(x=>(x._1._1,x._1._2,x._2.sum)).
          map(x=>(x._1,x._2,{
          if(x._1.equals(x._2)) 1.0-x._3/math.sqrt(d(x._1)*d(x._2)) else -x._3/math.sqrt(d(x._1)*d(x._2))
        }))
      }
    }
  }


  def calcular(data:RDD[RowVector])={
    val sc = data.sparkContext
    val eigen = new DistributedEigen().
      setMode("SA").
      setNumberOfEigenValue(numberOfEigen).
      run(data).sortBy(_._1)
    val eigenVec = eigen.map(_._2)

    eigen.foreach(x=>println(x._1))

    val dimension = eigenVec(0).length
    val kmeansData = sc.parallelize((0 until dimension).map(i=>(i, {
      var array = new Array[Double](numberOfEigen)
      for(j<- 0 until numberOfEigen){
        array(j) = eigenVec(j)(i)
      }
      Vectors.dense(array)
    }))).persist(StorageLevel.MEMORY_AND_DISK)
    data.unpersist()

    kMeans(kmeansData)
  }


  private def kMeans(v: RDD[(Int,mllib.linalg.Vector)]): RDD[(Int,Int)] = {
    val model = new KMeans()
      .setK(numberCluster)
      .setRuns(5)
      .setMaxIterations(100)
      .run(v.map(_._2))
    v.map(p =>(p._1,model.predict(p._2)))
  }

  def run(w:RDD[(Int,Int,Double)]):RDD[(Int,Int)]={
    val sc = w.sparkContext

    val lapMatrix = calcLaplacianMatrix(w).
      map(x=>(x._1,(x._2,x._3))).
      groupByKey().
      map(x=>RowVector(x._1,x._2.toArray)).
      setName("Laplacian Matrix").
      persist(StorageLevel.MEMORY_AND_DISK)


    calcular(lapMatrix)
  }

  def runWeightCut(w:RDD[(Int,Int,Double)], p:Array[Double]):RDD[(Int,Int)]={
    //    val p = weight.sortBy(_._1).collect().map(_._2)
    val lap = w.map(x=>(x._1,(x._2,x._3))).groupByKey().
      flatMap(x=>({
      val sum = x._2.map(_._2).sum
      x._2.map(t=>(x._1,t._1,{
        if(t._1 == x._1) sum-t._2 else -t._2
      }))
    })).
      map(x=>((x._1,x._2),x._3)).
      groupByKey().
      map(x=>(x._1._1,x._1._2,x._2.sum/math.sqrt(p(x._1._1)*p(x._1._2)))).
      map(x=>(x._1,(x._2,x._3))).
      groupByKey().
      map(x=>RowVector(x._1,x._2.toArray)).
      setName("Laplacian Matrix").
      persist(StorageLevel.MEMORY_AND_DISK)

    calcular(lap)
  }


  private def bipartiteNomalMode(w:RDD[(Int,Int,Double)]):BipartiteResult={
    val nRows = w.map(_._1).max()+1
    val nCols = w.map(_._2).max()+1
    val w1 = w.flatMap(x=>{
      Array((x._1,x._2+nRows,x._3),(x._2+nRows,x._1,x._3))
    })

    val maxRow = w1.map(_._1).max+1
    val maxCol = w1.map(_._2).max+1
    println(maxRow.toString+","+maxCol.toString+","+(nRows+nCols).toString)
    println(checkSymmetric(w1).toString)

    val result = run(w1).setName("Result").persist(StorageLevel.MEMORY_AND_DISK)
    val userCluster = result.filter(x=>x._1<nRows)
    val itemCluster = result.filter(x=>x._1>=nRows).map(x=>(x._1-nRows,x._2))
    BipartiteResult(userCluster,itemCluster)
  }

  private def bipartiteWeightMode(w:RDD[(Int,Int,Double)]):BipartiteResult={
    val nRows = w.map(_._1).max()+1
    val nCols = w.map(_._2).max()+1
    val w1 = w.flatMap(x=>{
      Array((x._1,x._2+nRows,x._3),(x._2+nRows,x._1,x._3))
    })

    val weightRow = 1.0/nRows
    val weightCol = 1.0/nCols
    val weight1 = (0 until nRows).map(i=>weightRow).toArray
    val weight2 = (0 until nCols).map(i=>weightCol).toArray
    val p = weight1.union(weight2)
    val result = runWeightCut(w1,p).setName("Result").persist(StorageLevel.MEMORY_AND_DISK)
    val userCluster = result.filter(x=>x._1<nRows)
    val itemCluster = result.filter(x=>x._1>=nRows).map(x=>(x._1-nRows,x._2))
    BipartiteResult(userCluster,itemCluster)

  }

  private def bipartiteHighLevel(w:RDD[(Int,Int,Double)]):BipartiteResult={

    val nRows = w.map(_._1).max()+1
    val nCols = w.map(_._2).max()+1
    val sc = w.sparkContext
    numberOfEigen = (math.log(numberCluster)/math.log(2)).toInt
    val sumColumn = w.map(x=>(x._2,x._3)).
      groupByKey().
      map(x=>(x._1,x._2.sum)).
      collect.
      sortBy(_._1).
      map(_._2)

    val globalSum = sc.broadcast(sumColumn)

    val lap = w.map(x=>(x._1,(x._2,x._3))).groupByKey().map(x=>(x._1,{
      val sum = x._2.map(_._2).sum
      //      Vectors.sparse(nCols,x._2.map{case(j,value)=>(j,{
      //        value/math.sqrt(sum*globalSum.value(j))
      //      })}.toArray)
      x._2.map{case(j,value)=>(j,{
        value/math.sqrt(sum*globalSum.value(j))
      })}

    })).map(x=>(RowVector(x._1,x._2.toArray))).setName("lap").
      persist(StorageLevel.MEMORY_AND_DISK)
    lap.count
    globalSum.unpersist()
    w.unpersist()

    //    val leftLap = lap.cartesian(lap).
    //      map{case(lap1,lap2)=>(lap1.row,(lap2.row,MathFunctions.sparseVectorProduct(lap1.value,lap2.value)))}.
    //      filter(_._2._2>1e-9).
    //      groupByKey().map(x=>RowVector(x._1,x._2.toArray))

    val indexedRowMatrix = new IndexedRowMatrix(lap.map(x=>IndexedRow(x.row.toLong,Vectors.sparse(nCols,x.value))))
      .toBlockMatrix()
    val leftLap = indexedRowMatrix.multiply(indexedRowMatrix.transpose).toIndexedRowMatrix().
      rows.map(x=>RowVector(x.index.toInt,x.vector.toArray.zipWithIndex.filter(_._1>1e-3).map(x=>(x._2,x._1))))

    /* val glomLap = lap.glom()
     val leftLap = glomLap.cartesian(glomLap).flatMap{case(g1,g2)=>{
       g1.flatMap(r1=>{
         g2.map(r2=>(r1.row,r2.row,MathFunctions.sparseVectorProduct(r1.value,r2.value)))
       })
     }}.filter(_._3>1e-3).map(x=>(x._1,(x._2,x._3))).groupByKey().map(x=>RowVector(x._1,x._2.toArray))*/


    BipartiteResult(calcular(leftLap),null)
  }


  /*
  * bipartite graph clustering
  * */
  def runBipartite(w:RDD[(Int,Int,Double)]):BipartiteResult={

    if(mode.equals("NCut")||mode.equals("RatioCut")||mode.equals("SymNCut")){
      bipartiteNomalMode(w)
    }else if(mode.equals("WeightCut")){
      bipartiteWeightMode(w)
    }else{
      bipartiteHighLevel(w)
    }

  }

  def checkSymmetric(w:RDD[(Int,Int,Double)]):Int={
    w.filter(x=> !(x._1 equals x._2)).map(x=>{
      if(x._1>x._2) ((x._1,x._2),x._3) else ((x._2,x._1),x._3)
    }).groupByKey().
      map(x=>(x._1,x._2.toArray)).
      map(x=>{
      if(x._2.length!=2){
        false
      }else if(x._2(0) equals x._2(1)){
        true
      }else {false}
    }).
      filter(x=> !x).
      count().
      toInt
  }

}





case class RowVector(row:Int, value:Array[(Int, Double)])

class DistributedEigen(var which:String,
                       var numberOfEigen:Int,
                       var maxIterations:Int,
                       var tolerance:Double,
                       var symmetric:Boolean
                        ) extends Serializable{



  def this(which:String, numberOfEigen:Int) = {
    this(which,numberOfEigen,300,1e-8,true)
  }
  def this() = this("LM",10)


  def setMode(m:String): this.type ={
    this.which = m
    this
  }

  def setNumberOfEigenValue(n:Int):this.type ={
    this.numberOfEigen = n
    this
  }

  def setMaxInteration(n:Int):this.type ={
    this.maxIterations = n
    this
  }

  def setTolerance(t:Double):this.type ={
    this.tolerance = t
    this
  }

  def setSymmetric(s:Boolean):this.type ={
    this.symmetric = s
    this
  }

  var n:Int=0

  def run(data:RDD[RowVector]): Array[(Double,Array[Double])]={
    if(data.getStorageLevel == StorageLevel.NONE){
      data.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val rowIDs = data.map(_.row).distinct()
    n = rowIDs.max()+1
    val nCols = data.flatMap(x=>x.value.map(_._1)).max+1
    require(n equals nCols,println("Error: nCols should equal to nRows"))

    val arpack = ARPACK.getInstance()

    val tolW = new doubleW(tolerance)
    val nev = new intW(numberOfEigen)
    val ncv = math.min(2 * numberOfEigen, n)
    val bmat = "I"
    var iparam = new Array[Int](11)
    iparam(0) = 1
    iparam(2) = maxIterations
    iparam(6) = 1

    var ido = new intW(0)
    var info = new intW(0)
    var resid = new Array[Double](n)
    var v = new Array[Double](n * ncv)
    var workd = new Array[Double](n * 3)
    var workl = new Array[Double](ncv * (ncv + 8))
    var ipntr = new Array[Int](11)

    arpack.dsaupd(ido, bmat, n, which, nev.`val`, tolW, resid, ncv, v, n, iparam, ipntr,
      workd, workl, workl.length, info)
    while (ido.`val` != 99) {
      if (ido.`val` != -1 && ido.`val` != 1) {
        throw new IllegalStateException("ARPACK returns ido = " + ido.`val` +
          " This flag is not compatible with Mode 1: A*x = lambda*x, A symmetric.")
      }

      val w = BDV(workd)
      val inputOffset = ipntr(0) - 1
      val outputOffset = ipntr(1) - 1
      val x = w.slice(inputOffset, inputOffset + n)
      val y = w.slice(outputOffset, outputOffset + n)


      y := BDV(matrixMultiply(data,x))
      arpack.dsaupd(ido, bmat, n, which, nev.`val`, tolW, resid, ncv, v, n, iparam, ipntr,
        workd, workl, workl.length, info)
    }

    if (info.`val` != 0) throw new IllegalStateException("info = " + info.`val`)

    val d = new Array[Double](nev.`val`)
    val select = new Array[Boolean](ncv)
    val z = java.util.Arrays.copyOfRange(v, 0, nev.`val` * n)

    arpack.dseupd(true, "A", select, d, z, n, 0.0, bmat, n, which, nev, tolerance, resid, ncv, v, n,
      iparam, ipntr, workd, workl, workl.length, info)

    val computed = iparam(4)

    var sortedU = new Array[(Double,Array[Double])](computed)
    for(i<- 0 until computed){
      val eigenVal = d(i)
      val eigenVec = java.util.Arrays.copyOfRange(z, i*n, i*n+n)
      sortedU(i)=(eigenVal,eigenVec)
    }
    sortedU.sortBy(-_._1)
  }



  def matrixMultiply(matrix:RDD[RowVector], vector: Array[Double]):Array[Double]={
    val sc = matrix.sparkContext
    val global = sc.broadcast(vector)
    val calc = matrix.map(rv=> (rv.row,{
      rv.value.map(x=>global.value(x._1)*x._2).sum
    })).collect().
      sortBy(_._1).
      map(_._2)
    global.unpersist()
    calc
  }

  def matrixMultiply(matrix:RDD[RowVector], vector: BDV[Double]):Array[Double]={
    matrixMultiply(matrix, vector.toArray)

  }

  def testPrecision(data:RDD[RowVector], eigens:Array[(Double,Array[Double])]):Array[Double]={
    val sc = data.sparkContext
    eigens.map{case(eigenVal,eigenVec)=>{
      val ei = sc.parallelize((0 until n).map(i=>{
        ((i,i),-eigenVal)
      }))
      val r = data.flatMap(x=>x.value.map(xx=>((x.row,xx._1),xx._2))).union(ei).
        groupByKey().
        map(x=>(x._1._1,(x._1._2,x._2.sum))).
        groupByKey().
        map(x=>RowVector(x._1,x._2.toArray))
      matrixMultiply(r,eigenVec).reduce((a,b)=> math.abs(a)+math.abs(b))
    }}
  }

  def checkSymmetric(data:RDD[RowVector]):Int={
    data.flatMap(x=>x.value.map(xx=>(x.row,xx._1,xx._2))).
      filter(x=> x._1!=x._2).map(x=>{
      if(x._1>x._2) ((x._1,x._2),x._3) else ((x._2,x._1),x._3)
    }).groupByKey().map(x=>(x._1,x._2.toArray)).map(x=>{
      if(x._2.length!=2){
        false
      }else if(x._2(0)!=x._2(1)){
        false
      }else {true}
    }).filter(x=> !x).count.toInt
  }

}
