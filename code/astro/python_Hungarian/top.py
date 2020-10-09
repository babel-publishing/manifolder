
import numpy as np                                                                                                                                  
cl1 = np.array([1, 1, 2, 3, 3, 4, 4, 4, 2])                                                                                                         
cl2 = np.array([2, 2, 3, 1, 1, 4, 4, 4, 3])                                                                                                         
import best_map 
a = best_map.best_map(cl1, cl2) 
print(a)
