#################################################################
#                                                               #
#   FUNCTIONS FOR EXPRESSING CSP FORMULAS AS TENSOR NETWORKS    #
#   ========================================================    #
#   ( tensorcsp.py )                                            #
#   first instance: 20170726                                    #
#   written by Stefanos Kourtis ( kourtis@bu.edu )              #
#                                                               #
#   Routines for encoding constraint satisfaction problems as   #
#   tensor networks and basic tools for contraction. Contains   #
#   utilities for CNF formulas, as well as basic reading and    #
#   writing of CNF instances in DIMACS format.                  #
#                                                               #
#   DEPENDENCIES: numpy, igraph, grut.py                        #
#                                                               #
#################################################################

import sys                      # Import system-specific library
sys.dont_write_bytecode = True  # Do not write compiled bytecode
from numpy  import *            # Use numpy for linear algebra
from igraph import *            # Import igraph
from grut   import *            # Import graph utilities

# Boolean gate functions; i holds input bits, m is negation mask
def oror(i,m=0): return int(sum(array(list(binary_repr(bitwise_xor(i,m))),int))>0)
def xorxor(i,m=0): return int(sum(array(list(binary_repr(bitwise_xor(i,m))),int))%2>0)

def var_tensor(l=3,q=2,dtype=int):
    """ Return variable tensor of rank l with entries over
        domain of dimension q. Also called COPY tensor. """
    t = zeros([q]*l,dtype)
    t[diag_indices(q,l)] = 1
    return array(t,dtype)

def clause_tensor(l,q=2,g=oror,m=0,dtype=int):
    """ Return tensorization of the truth table of gate g.
        Generally, g is a l-ary relation in the constraint
        language of a (weighted) CSP and this function
        returns a tensor representation of the relation g. """
    d = [q]*l
    t = zeros(d,dtype)
    for i in range(q**l): t[unravel_index(i,d)] = g(i,m)
    return array(t,dtype)

def cnf_read(fs,sort_clauses=True,read_xors=False):
    """ Read CNF formula(s) from file(s) in DIMACS format.
        The read_xors flag is used to read extended DIMACS
        format of cryptominisat2 supporting XOR clauses. """
    if ( type(fs) == str ): fs = [fs]
    xs = []
    cs = []
    for f in fs:
        c = []
        x = []
        for l in open(f):
            if ( l[0] == 'c' or l[0] == 'p' ): continue
            if ( l[0] == 'x' ):
                l = l[1:]
                x.append(1)
            else:
                x.append(0)
            vs = array([int(i) for i in l.split()[:-1]])
            if ( sort_clauses ): vs = vs[argsort(abs(vs))]
            c.append(vs)
        if ( all(array([len(r) for r in c])==len(c[0])) ): c = array(c)
        cs.append(c)
        xs.append(x)
    if ( len(cs) == 1 ): cs = cs[0]
    if ( read_xors ): return cs,array(xs)
    return cs

def cnf_write(c,filename,xs=None):
    """ Write CNF formula c to file in DIMACS format. """
    nv = cnf_nvar(c)
    nc = len(c)
    xs = [' ']*nc if ( xs == None ) else array([' ','x'])[xs]
    f  = open(filename,'w')
    f.write('p cnf '+str(nv)+' '+str(nc)+'\n')
    for i in range(nc): f.write(xs[i]+' '.join([str(j) for j in c[i]]) + ' 0\n')
    f.close()

def cnf_negmask(c):
    """ Negation mask of CNF formula c. """
    return [int(''.join(s),2) for s in [array(array(array(a)<0,int),str) for a in c]]

def cnf_nvar(c):
    """ Number of variables in CNF formula c. """
    return array([abs(array(list(a))).max() for a in c]).max()

def cnf_graph(c):
    """ Returns bipartite graph object corresponding to
        CNF formula c. Components represent variables
        and clauses and edges connect variables to the
        clauses they participate in. """
    nc = len(c)
    nv = cnf_nvar(c)
    a  = zeros([nc,nv],int)
    b  = zeros([nc+nv,nc+nv],int) # Biadjacency matrix
    for i,r in enumerate(c): a[i][abs(array(r))-1] = 1
    b[nv:,:nv] = a
    b = b+b.T
    g = Graph.Adjacency(b.astype(bool).tolist(),mode=ADJ_UNDIRECTED)
    return g

def cnf_tn(c,q=2,gate=oror,dtype=int):
    """ Returns tensor network for CNF formula c. The
        tensor network is defined on the graph returned
        by cnf_graph(). """
    nv = cnf_nvar(c)
    nm = cnf_negmask(c)
    g  = cnf_graph(c)           # Get CNF graph
    al = g.get_adjlist()        # Adjacency list of g
    _,tp = g.is_bipartite(True) # Get vertex types
    # NB: assume variables always come before clauses
    cv = arange(g.vcount())[nonzero(tp)[0]]
    # Build tensor network; variable tensors first
    tn = [var_tensor(len(l),q,dtype=dtype) for l in al[:nv]]
    for n,i in enumerate(cv):
        nb = len(al[i])
        ct = clause_tensor(nb,q,gate,m=nm[n],dtype=dtype)
        tn.append(ct)
    return tn

def cnf_tngraph(c,q=2,gate=oror,dtype=int):
    """ Graph object including tensor representation of
        CNF formula c. Each vertex stores (a) a list of
        unique edge indices incident to it, and (b) the
        truth tensor corresponding to variable / clause
        as a list attribute named 'attr'. """
    nc = len(c)
    nv = cnf_nvar(c)
    g  = cnf_graph(c)
    tn = cnf_tn(c,q,gate=gate,dtype=dtype)
    # Use initial edge indices for unique bond indexing
    # throughout contraction sequence
    il = g.get_inclist()
    for i in range(g.vcount()): g.vs[i]["attr"] = [il[i],tn[i]]
    return g

def attr_contract(d):
    """ Contraction function for vertex attributes, to be
        used with graph edge contraction. The input is a
        a 2-element list d containing the attributes of
        the two vertices, each of which is also a list of
        an incidence list and a tensor. This function
        finds common indices in the two incidence lists,
        contracts corresponding tensor dimensions, then
        concatenates incidence lists into a new one. The
        purpose of this function is to be passed as an
        argument to igraph's contract_vertices(). """
    if ( len(d) == 0 ): return [[],[]]
    if ( len(d) == 1 ): return d[0]
    i1,i2 = d[0][0],d[1][0]
    t1,t2 = d[0][1],d[1][1]
    ce = intersect1d(i1,i2)
    d1 = [i1.index(i) for i in ce]
    d2 = [i2.index(i) for i in ce]
    for i in sort(d1)[::-1]: i1.pop(i)
    for i in sort(d2)[::-1]: i2.pop(i)
    t = tensordot(t1,t2,[d1,d2])
    return [i1+i2,t] # CAUTION! inclist *NOT* sorted

