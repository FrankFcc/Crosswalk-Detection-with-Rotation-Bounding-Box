import math
import numpy as np
def num(*args, **arg):
    pass

print = num
def dis(x1,y1,x2,y2):
  return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def isRectangle(p1, p2, p3, p4):
    x_c = (p1[0] + p2[0] + p3[0] + p4[0])/4
    y_c = (p1[1] + p2[1] + p3[1] + p4[1])/4
    d1 = dis(p1[0], p1[1], x_c,y_c)
    d2 = dis(p2[0], p2[1], x_c,y_c)
    d3 = dis(p3[0], p3[1], x_c,y_c)
    d4 = dis(p4[0], p4[1], x_c,y_c)
    print(d1,d2,d3,d4)
    return d1 == d2 and d1 == d3 and d1 == d4
def vecMuti(u, v):
  return u[0]*v[0] + u[1]*v[1]

def vecMuti_all(u0,u1,v0,v1):
  return u0*v0 + u1*v1

def m(v):
  return math.sqrt(v[0] ** 2 + v[1] ** 2)

def p(u, v):
  # v on u
  return vecMuti(u,v)/ m(u)

def solve(a, b, c):
  d = (b**2) - 4*a*c
  # print(a,b,c,d)

  return  (-b-math.sqrt(d))/(2*a),(-b+math.sqrt(d))/(2*a)

def solve_all(a, b, c):
  d = (b**2) - 4*a*c
  return (-b-np.sqrt(d))/(2*a+1**(-7)),(-b+np.sqrt(d))/(2*a+1**(-7))

def dis_all(x1,y1,x2,y2):
  return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def m_all(v0, v1):
  return np.sqrt(v0 ** 2 + v1 ** 2)

def p_all(u0, u1, v0, v1):
  t = u0 * v0 + u1 * v1
  return t / m_all(u0, u1) 

def convert_all(x1, y1, x2, y2, x3, y3, x4, y4, cx, cy):
  l1 = dis_all(x1, y1, x2, y2)
  l2 = dis_all(x2, y2, x3, y3)
  # print('dis',l1,l2)
  dx = np.zeros_like(x1)
  dy = np.zeros_like(x1)
  ex = np.zeros_like(x1)
  ey = np.zeros_like(x1)

  dx[l1<l2] = (x1 - cx)[l1<l2]
  dy[l1<l2] = (y1 - cy)[l1<l2]
  ex[l1<l2] = (x2 - cx)[l1<l2]
  ey[l1<l2] = (y2 - cy)[l1<l2]

  dx[l1>l2] = (x2 - cx)[l1>l2]
  dy[l1>l2] = (y2 - cy)[l1>l2]
  ex[l1>l2] = (x3 - cx)[l1>l2]
  ey[l1>l2] = (y3 - cy)[l1>l2]

  dx[l1==l2] = (x2+1 - cx)[l1==l2]
  dy[l1==l2] = (y2+1 - cy)[l1==l2]
  ex[l1==l2] = (x3+1 - cx)[l1==l2]
  ey[l1==l2] = (y3+1 - cy)[l1==l2]

  pp = p_all(ex, ey, dx, dy) / m_all(dx, dy)
  s = np.sign(dx) * np.sign(dy)
  s[s==-1] = 0
  return (cx, cy, np.abs(dx), np.abs(dy), s, pp )

def convert(points, cx, cy):
  x1,y1 = points[0]
  x2,y2 = points[1]
  x3,y3 = points[2]
  x4,y4 = points[3]

  l1 = dis(x1,y1,x2,y2)
  l2 = dis(x2,y2,x3,y3)
  print(l1,l2)

  if l1 < l2:
    dx = (x1 - cx)
    dy = (y1 - cy)
    ex = (x2 - cx)
    ey = (y2 - cy)
  elif l2 < l1:
    dx = (x2 - cx)
    dy = (y2 - cy)
    ex = (x3 - cx)
    ey = (y3 - cy)
  else:
    points = [(x1+1,y1+1), (x2,y2), (x3+1, y3+1), (x4, y4)]
    return convert(points, cx, cy)
    # raise Exception("cant be rect")
    # + 1
  pp = p((ex, ey), (dx, dy)) / m((dx, dy))
  s = (dx / abs(dx)) * (dy / abs(dy))
   
  return (cx, cy, abs(dx), abs(dy), s, pp )

def wh2fourpoint(cx, cy, w, h):
  pass

def deconvert_all(cx, cy , adx, ady, s, pp):
  # print(cx, cy , adx, ady, s, pp)

  s[s<0.5] = -1
  s[s>=0.5] = 1

  dx = adx * s
  dy = ady

  dx2 = dx * -1
  dy2 = dy * -1

  
  tdx = dx + cx
  tdy = dy + cy
  tdx2 = dx2 + cx
  tdy2 = dy2 + cy

  k = dx**2 + dy**2
  a = k
  b = -2 * pp * k * dy
  c = (pp**2) * (k**2) - (dx**2)*k
  d = (b**2) - 4*a*c
  a[d<0] = 0
  b[d<0] = 0
  c[d<0] = 0
  t1,t2 = solve_all(a, b, c)
  tx = (pp*k - dy*t1) / dx
  tx2 = (pp*k - dy*t2) / dx
  ex = np.zeros_like(tx)
  ey = np.zeros_like(t1)

  flag = vecMuti((tx, t1), (dx, dy))
  ex[flag>=0] = tx2[flag>=0]
  ey[flag>=0] = t2[flag>=0]

  ex[flag<0] = tx[flag<0]
  ey[flag<0] = t1[flag<0]

  ex2 = -ex
  ey2 = -ey
  
  tex = ex + cx
  tey = ey + cy
  print(tx2+cx,t2+cy,tx+cx,t1+cy)

  tex2 = ex2 + cx
  tey2 = ey2 + cy
  return tdx,tdy,tex,tey,tdx2,tdy2,tex2,tey2


def deconvert(cx, cy, adx, ady, s, pp):
  w = math.sqrt(2 * (m((adx, ady)) ** 2) - 2 * pp * (m((adx, ady))**2) )
  h = math.sqrt(2 * (m((adx, ady)) ** 2) + 2 * pp * (m((adx, ady))**2) )

  dx = adx * s
  dy = ady

  dx2 = dx * -1
  dy2 = dy * -1

  
  tdx = dx + cx
  tdy = dy + cy
  tdx2 = dx2 + cx
  tdy2 = dy2 + cy
  
  k = dx**2 + dy**2
  t1,t2 = solve(k, -2 * pp * k * dy, (pp**2) * (k**2) - (dx**2)*k)
  tx = (pp*k - dy*t1) / dx
  tx2 = (pp*k - dy*t2) / dx
  # print('x1x2',tx, t1, tx2, t2, dx, dy, k)

  if vecMuti((tx, t1), (dx, dy)) > 0:
    ex = tx
    ey = t1
  else:
    ex = tx2
    ey = t2
  
  ex2 = -ex
  ey2 = -ey
  
  tex = ex + cx
  tey = ey + cy

  tex2 = ex2 + cx
  tey2 = ey2 + cy


  return tdx,tdy,tdx2,tdy2,tex,tey,tex2,tey2, w, h



def test():
  p1 = (1271.70953668,599.55027227)
  p2 = (1331.29046332,688.44972773)
  p3 = (1314.03796056,591.98029657)
  p4 = (1288.96203944,696.01970343)

  print(isRectangle(p1,p2,p3,p4))
  (cx, cy, ddx, ddy, s, pp ) = convert([p1,p2,p3,p4], 1301.5,644)

  print(cx, cy, ddx, ddy, s, pp )
  print(deconvert(cx, cy, ddx, ddy, s, pp ))

def test_all():
  test()
  x1 = np.array([[1271.70953668]])
  y1 = np.array([[599.55027227]])

  x2 = np.array([[1288.96203944]])
  y2 = np.array([[696.01970343]])

  x3 = np.array([[1331.29046332]])
  y3 = np.array([[688.44972773]])

  x4 = np.array([[1314.03796056]])
  y4 = np.array([[591.98029657]])
  cx = np.array([[1301.5]])
  cy = np.array([[644]])
  print(convert_all(x1, y1, x2, y2, x3, y3, x4, y4, cx, cy))
  
  print(deconvert_all(*convert_all(x1, y1, x2, y2, x3, y3, x4, y4, cx, cy)))

if __name__ == '__main__':
  test_all()