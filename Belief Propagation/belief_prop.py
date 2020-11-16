"""
Refer - http://6.869.csail.mit.edu/fa13/lectures/slideNotesCh7rev.pdf
        http://helper.ipam.ucla.edu/publications/gss2013/gss2013_11344.pdf

This module finds the beliefs of nodes 1,2,4,3,5 which do not have a label.
                    (1)    1
                    4 ---- 7'
                   /
    0'    (0)     /    (0)
    1 -- -2 --- 3 ---- 5
          |     (1)
          |
          6' 0

    phi(2,6') = phi(4,7') = [[1, 0.1], [0.1, 1]]
    sci(1,2) = sci(3,4) = [[1, 0.9], [0.9, 1]]
    sci(2,3) = sci(3,5) = [[0.1, 1], [1, 0.1]]
    6' is 0
    7' is 1
    Compute beliefs of 1, 2, 3, 4, 5
    m74, m53, m62, m12

    m23 = sci_2_3 * m12 . m62
    m34 = sci_3_4 * m23. m53
    m43 = sci_3_4*m74
    m32 =sci_2_3*m43.m53
    m35 = sci_3_5*m43.m23
    m21 = sci_2_1*m62.m32
"""
import numpy as np

phi_2_6, phi_4_7 = np.array([[1, 0.1], [0.1, 1]]), np.array([[1, 0.1], [0.1, 1]])
sci_1_2, sci_3_4 = np.array([[1, 0.9], [0.9, 1]]), np.array([[1, 0.9], [0.9, 1]])
sci_2_3, sci_3_5 = np.array([[0.1, 1], [1, 0.1]]), np.array([[0.1, 1], [1, 0.1]])

m12 = np.sum(sci_1_2, axis=1).reshape(2, 1)
m12 = m12 / np.max(m12)


delta_y6 = np.array([[1, 0]])
m62 = np.matmul(delta_y6, phi_2_6).transpose()

m53 = np.sum(sci_3_5, axis=1).reshape(2, 1)
m53 = m53 / np.max(m53)

delta_y7 = np.array([[0, 1]])
m74 = np.matmul(delta_y7, phi_4_7).transpose()


m23 = np.matmul(sci_2_3, np.multiply(m12, m62))
m34 = np.matmul(sci_3_4, np.multiply(m23, m53))
m43 = np.matmul(sci_3_4, m74)
m32 = np.matmul(sci_2_3, np.multiply(m43, m53))
m35 = np.matmul(sci_3_5, np.multiply(m43, m23))
m21 = np.matmul(sci_1_2, np.multiply(m62, m32))


b1_x1 = m21 / np.sum(m21)
b2_x2 = np.multiply(m62, np.multiply(m12, m32))
b2_x2 = b2_x2 / np.sum(b2_x2)
b4_x4 = np.multiply(m34, m74)
b4_x4 = b4_x4/np.sum(b4_x4)
b3_x3 = np.multiply(m53, np.multiply(m43, m23))
b3_x3 = b3_x3/np.sum(b3_x3)
b5_x5 = m35 / np.sum(m35)

print(f"b1_x1 : {b1_x1}")
print(f"b2_x2 : {b2_x2}")
print(f"b3_x3 : {b3_x3}")
print(f"b4_x4 : {b4_x4}")
print(f"b5_x5 : {b5_x5}")


