#################################################################
#                                                               #
#   GRAPH UTILITIES                                             #
#   ===============                                             #
#   ( grut.py )                                                 #
#   first instance: 20170726                                    #
#   written by Stefanos Kourtis ( kourtis@bu.edu )              #
#                                                               #
#   Auxiliary routines for evaluation of graph properties and   #
#   graph operations based on the igraph library. Partitions    #
#   are obtained using the METIS library, which needs to be     #
#   installed for the corresponding functions to work.          #
#                                                               #
#   DEPENDENCIES: numpy, igraph, metis                          #
#                                                               #
#################################################################

import sys
sys.dont_write_bytecode = True  # Do not write compiled bytecode
from numpy import *
from numpy.linalg import eigh
from copy import deepcopy
from igraph import *
try:
    from metis import part_graph
except ImportError:
    pass

def adjmat(g):
    """ Adjacency matrix of graph object g as ndarray. """
    return array(g.get_adjacency().data)

def adjlist2adjmat(a):
    """ Convert adjacency list to adjacency matrix. """
    n = array(sum(a+[[]])).max()+1
    m = zeros([n,n],int)
    for i,r in enumerate(a):
        for l in r: m[i,l] = m[i,l] + 1
    return m

def get_cluster_vids(membership):
    """ Return vertex indices grouped according to
        graph clustering as encoded in membership vector. """
    nv = len(membership)
    nc = array(membership).max()+1
    vi = []
    for i in range(nc):
        vi = vi + [list((array(membership)==i).nonzero()[0])]
    return vi

def get_cluster_eids(membership,g):
    """ Return edge indices grouped according to graph
        clustering as encoded in membership vector. """
    vi = get_cluster_vids(membership)
    ei = []
    for v in vi: ei = ei + [g.es.select(_within=v).indices]
    return ei

def get_bipartition_eids(membership,g):
    """ Given a bipartition encoded in a membership
        vector, return the indices of the edges that form
        the edge separator between components. """
    m = array(membership,int)
    s1 = nonzero(m==0)[0]
    s2 = nonzero(m>0)[0]
    return g.es.select(_between=[s1,s2]).indices

def bipartition_width(membership,g):
    """ Return bipartition width from membership vector. """
    return len(get_bipartition_eids(membership,g))

def metis_bipartition(g,n=2):
    """ Perform METIS bipartition. Do n cuts and choose the best. """
    al = g.get_adjlist()
    _,m = part_graph(al,ncuts=n,recursive=False,contig=True,minconn=True)
    return array(m)

def metis_kway(g,k):
    """ Perform METIS k-way partition. """
    al = g.get_adjlist()
    _,m = part_graph(al,k,recursive=False,contig=True,minconn=True)
    return m

# Fiedler vector does not guarantee contiguous partitions.
# This function is here for testing purposes only.
def fiedler_bipartition(g):
    """ Perform bipartition based on the Fiedler vector. """
    l = array(g.laplacian())
    _,v = eigh(l)
    return v[:,1]<=0

def recursive_bipartition(g,fbipart=metis_bipartition):
    """ Build separator hierarchy using recursive bipartition.
        Returns a dendrogram merge sequence. """
    nv = g.vcount()
    cg = deepcopy(g)
    cg.vs["name"] = range(nv)
    sg = [cg]
    tr = []
    im = 2*nv
    for i in range(nv):
        st = []
        for j,s in enumerate(sg):
            if ( s.vcount() == 1 ): continue
            fb = fbipart(s)
            s1 = nonzero(fb==0)[0]
            s2 = nonzero(fb>0)[0]
            # METIS often refuses to partition very small graphs
            # so "peel off" least connected vertex instead
            while ( len(s1)*len(s2) == 0 ):
                fb = zeros(s.vcount(),int)
                fb[array(s.degree()).argmin()] = 1
                s1 = nonzero(array(fb)==0)[0]
                s2 = nonzero(array(fb)>0)[0]
            g1 = s.subgraph(s1)
            g2 = s.subgraph(s2)
            st.append(g1)
            st.append(g2)
            if ( len(s1) == 1 ):
                i1 = g1.vs[0]["name"]
            else:
                im = im-1
                i1 = im
            if ( len(s2) == 1 ):
                i2 = g2.vs[0]["name"]
            else:
                im = im-1
                i2 = im
            tr = tr + [array([i1,i2])]
        sg = st
        if ( len(st) == 0 ): break
    tr = array(tr[::-1])
    th = tr[tr>=nv].min()
    df = th-nv
    tr[tr>=nv] = tr[tr>=nv] - df
    return tr

def find_cheapest_edge(g,subset=None):
    """ Return index of edge whose contraction leads to the
        graph minor of g with the lowest maximum degree.
        Optionally restrict the search to a subset. """
    dg = array(g.vs.degree(),int)
    el = array(g.get_edgelist())
    if ( subset == None ): subset = arange(g.ecount())
    ms = array(g.count_multiple(),int)[subset]
    ds = zeros(len(subset))
    for i,ee in enumerate(subset):
        ds[i] = (dg[el[ee][0]] - ms[i])*(dg[el[ee][1]] - ms[i])
    ie = subset[ds.argmin()]
    return ie

def contract_edge(g,ie,combine_attrs=None,overwrite="higher"):
    """ Perform edge contraction with user-specified function
        to combine vertex attributes. The contracted vertex
        can either overwrite existing vertices or be added to
        the graph. """
    el = g.get_edgelist()
    nv = g.vcount()
    ne = g.ecount()
    ee = el[ie]
    ar = arange(nv)
    if ( overwrite == "higher" ):
        ar[ee[0]] = ee[1] # Overwrite vertex with higher index
    elif ( overwrite == "lower" ):
        ar[ee[1]] = ee[0] # Overwrite vertex with lower index
    # Overwrite vertex with specified index
    elif ( issubdtype(type(overwrite),integer) ):
        if ( overwrite == ee[0] ):
            ar[ee[1]] = ee[0]
        elif ( overwrite == ee[1] ):
            ar[ee[0]] = ee[1]
        else:
            ar[ee[0]] = overwrite
            ar[ee[1]] = overwrite
    elif ( (overwrite == "none") or (overwrite == None) ):
        ar[ee[0]] = nv    # Contract into new vertex
        ar[ee[1]] = nv
    while ( ie >= 0 ):
        g.delete_edges(ie)
        ie = g.get_eid(ee[0],ee[1],error=False)
    g.contract_vertices(ar,combine_attrs=combine_attrs)

def contract_greedy(g,n=0,fsize=Graph.maxdegree,combine_attrs=None):
    """ Greedy contraction algorithm. """
    nv = g.vcount()
    bs = [fsize(g)]
    # NB: it is annoyingly hard to deep-copy igraph graphs!
    cg = deepcopy(g)
    if ( n == 0 ): n = 100000
    while ( cg.ecount() > 0 and n > 0 ):
        ie = find_cheapest_edge(cg)
        contract_edge(cg,ie,combine_attrs=combine_attrs)
        bs = bs + [fsize(cg)]
        n = n - 1
    bs = array(bs)
    vd = nonzero(array(cg.degree())==0)[0]
    if ( len(vd) == cg.vcount() ): vd = vd[:-1]
    cg.delete_vertices(vd)
    return bs,cg

def contract_dendrogram(g,merges,fsize=Graph.maxdegree,combine_attrs=None,stop=-1):
    """ Contract graph according to merge sequence. """
    merges = array(merges,int)
    ne = g.ecount()
    bs = [fsize(g)]
    # NB: it is annoyingly hard to deep-copy igraph graphs!
    cg = deepcopy(g)
    for i,m in enumerate(merges):
        ie = cg.get_eid(m[0],m[1])
        contract_edge(cg,ie,combine_attrs=combine_attrs,overwrite="none")
        bs = bs + [fsize(cg)]
        if ( i == stop-1 ): break
    bs = array(bs)
    vd = nonzero(array(cg.degree())==0)[0]
    if ( len(vd) == cg.vcount() ): vd = vd[:-1]
    cg.delete_vertices(vd)
    return bs,cg
