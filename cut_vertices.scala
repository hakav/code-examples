package cutVertices

import collection.mutable.{ArrayBuffer,Queue}

/**
 * A simple immutable class for undirected graphs.
 * The vertices are integers from 0 to nofVertices-1.
 * The edges are of form (vertex1, vertex2).
 * Self-loops and parallel edges between two vertices are not supported.
 */
class Graph(val nofVertices: Int, edges: Seq[(Int,Int)]):
  require(nofVertices > 0)

  /**
   * neighbours(u) is the list of vertices v such {u,v} is an edge.
   */
  val neighbours = Array.fill[ArrayBuffer[Int]](nofVertices)(ArrayBuffer[Int]())

  /* Validate the input and build the neighbours data structure.
   * Duplicate edges are discarded. */
  private def init(edges: Seq[(Int,Int)]) =
    val seen = collection.mutable.HashSet[(Int,Int)]()
    for (vertex1, vertex2) <- edges if !seen((vertex1, vertex2)) do
      require(0 <= vertex1 && vertex1 < nofVertices)
      require(0 <= vertex2 && vertex2 < nofVertices)
      neighbours(vertex1) += vertex2
      neighbours(vertex2) += vertex1
      seen((vertex1,vertex2)) = true
      seen((vertex2,vertex1)) = true
  end init
  init(edges)


  /**
   * Get the degree of a vertex.
   */
  def degree(v: Int): Int =
    require(0 <= v && v < nofVertices)
    neighbours(v).length


  /**
   * Get the maximum degree.
   * A constant time operation after the initialization phase.
   */
  val maxDegree =
    (0 until nofVertices).map(v => degree(v)).max


  /* As the graph data structure here is immutable and
   * we use connectedness as a requirement in many places,
   * compute this information once in the beginning. */
  val isConnected: Boolean =
    if nofVertices == 0 then true
    else
      val seen = new Array[Boolean](nofVertices)
      val q = new Queue[Int]()
      q.enqueue(0)
      seen(0) = true
      var nofSeen = 1
      while q.nonEmpty do
        val v: Int = q.dequeue()
        for w <- neighbours(v) if !seen(w) do
          q.enqueue(w)
          seen(w) = true
          nofSeen += 1
      nofSeen == nofVertices
  end isConnected


  /**
   * Find the cut vertices in the graph.
   * A straightforward, O(|V|(|V|+|E|)) time solution:
   * for each vertex, virtually delete the vertex from the graph and
   * check whether the resulting graph has more connected components.
   */
  def cutVerticesSlow: Set[Int] =
    val reachable = new Array[Boolean](nofVertices)
    val queue = new Queue[Int]()

    // Mark the vertices that are reachable from 'source'
    // without visiting 'exclude' in the array 'reachable'
    def reach(source: Int, exclude: Int) =
      require(0 <= source && source < nofVertices)
      require(0 <= exclude && exclude < nofVertices)
      require(source != exclude)
      // Reset the 'reachable' array and the search queue
      for v <- 0 until nofVertices do reachable(v) = false
      queue.clear()
      queue.enqueue(source)
      reachable(source) = true
      while queue.nonEmpty do
        val v = queue.dequeue()
        for w <- neighbours(v) if w != exclude do
          if !reachable(w) then
            reachable(w) = true
            queue.enqueue(w)
    end reach

    val cutVertices = ArrayBuffer[Int]()
    for v <- 0 until nofVertices if neighbours(v).nonEmpty do
      reach(neighbours(v).head, v)
      if !neighbours(v).forall(w => reachable(w)) then
        cutVertices += v

    cutVertices.toSet
  end cutVerticesSlow


  /**
   * Find the cut vertices in the graph.
   * A faster, O(|V|+|E|) time algorithm.
   * http://en.wikipedia.org/wiki/Biconnected_component
   */
  def cutVertices: Set[Int] =
    val visited = Array.ofDim[Boolean](nofVertices)
    var parent = collection.mutable.HashMap[Int, Int]()
    var low = Array.ofDim[Int](nofVertices)
    var depth = Array.ofDim[Int](nofVertices)
    var cuts = Array[Int]()

    def search(i: Int, d: Int): Unit = 
      visited(i) = true
      depth(i) = d
      low(i) = d
      var childCount = 0
      var isArticulation = false
      for (ni <- neighbours(i)) do
        if !visited(ni) then
          parent += ((ni, i))
          search(ni, d + 1)
          childCount += 1
          if low(ni) >= depth(i) then
            isArticulation = true
          low(i) = Math.min(low(i), low(ni)).toInt
        else if parent.contains(i) && parent(i) != ni then
          low(i) = Math.min(low(i), depth(ni)).toInt
      if (parent.contains(i) && isArticulation) || (!parent.contains(i) && childCount > 1) then
        cuts = cuts :+ i

    search(0, 0)
    cuts.toSet
  end cutVertices




end Graph 
