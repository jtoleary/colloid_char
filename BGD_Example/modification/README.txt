In the neighborlist.py, please add following lines below line 379:
##################################################################
        #### RM added 10-28-2020 Lehigh
          if snap.tag == 'same':
            if self.clustering:
                nn = self.filterNeighbors(idx,idx,nl,snap)
                if idx <snap.total_na:
                   nn = nn[nn<snap.total_na]
                else:
                   nn = nn[nn>=snap.total_na]
            else:
                nn = nl[idx]
            all_neighbors.append(np.array(nn,dtype=np.int))
          elif snap.tag == 'diff':
            if self.clustering:
                nn = self.filterNeighbors(idx,idx,nl,snap)
                if idx <snap.total_na:
                   nn = nn[nn>=snap.total_na]
                   nn=np.append(nn,idx)
                else:
                   nn = nn[nn<snap.total_na]
                   nn=np.append(nn,idx)
            else:
                nn = nl[idx]
            all_neighbors.append(np.array(nn,dtype=np.int))
          elif snap.tag == 'all':
            if self.clustering:
                nn = self.filterNeighbors(idx,idx,nl,snap)
            else:
                nn = nl[idx]
            all_neighbors.append(np.array(nn,dtype=np.int))
           #### RM added 10-28-2020 Lehigh

In the nga.py, please add following entry for particle counts below line 49:
############################################################################
        #self.total_na = None #RM added
        if self.xyz is not None:
            self.N = len(self.xyz)
        else:
            self.N = None
        # RM added
