import pandas as pd
import numpy as np
import scipy.integrate as integrate

def mortality_rate(x):
  return 0.00022 + 2.7 * (10**-6) * (1.124**x)

def select_mortality_rate(s, x):
  return 0.9 ** (2 - s) * mortality_rate(x + s)

v = 1.055 ** -1
v2 = 1.113025 ** -1

df = pd.DataFrame(range(121), columns=['x'])

# p_x
for i in range(0, 120): df.loc[i, 'p_x'] = np.exp(-integrate.quad(mortality_rate, i, i + 1)[0])
df.loc[120, 'p_x'] = 0

# q_x
df['q_x'] = 1 - df['p_x']

# ..a_x
df.loc[120, 'a_x'] = 1
for i in reversed(range(120)): df.loc[i, 'a_x'] = 1 + v * df.loc[i, 'p_x'] * df.loc[i + 1, 'a_x']

# A_x
df['A_x'] = 1 - (1 - v) * df['a_x']

# 2A_x
df.loc[120, '2a_x'] = 1
for i in reversed(range(120)): df.loc[i, '2a_x'] = 1 + v2 * df.loc[i, 'p_x'] * df.loc[i + 1, '2a_x']
df['2A_x'] = 1 - (1 - v2) * df['2a_x']

# Select mortality helpers
for i in range(120): df.loc[i, '1p[x]'] = np.exp(-integrate.quad(select_mortality_rate, 0, 1, args=(i))[0])
df.loc[120, '1p[x]'] = 0
df['1q[x]'] = 1 - df['1p[x]']
for i in range(119): df.loc[i, '2p[x]'] = np.exp(-integrate.quad(select_mortality_rate, 0, 2, args=(i))[0])
df.loc[119, '2p[x]'] = 0
df.loc[120, '2p[x]'] = 0
for i in range(119): df.loc[i, '1q[x+1]'] = 1 - np.exp(-integrate.quad(select_mortality_rate, 1, 2, args=(i))[0])
df.loc[119, '1q[x+1]'] = 1
df.loc[120, '1q[x+1]'] = 1

# a_[x]
for i in range(121):
  res = 1
  if i < 119: res += (df.loc[i, '2p[x]'] * v ** 2) * df.loc[i + 2, 'a_x']
  if i < 120: res += (df.loc[i, '1p[x]'] * v)
  df.loc[i, 'a_[x]'] = res

# A_[x]
df['A_[x]'] = 1 - (1 - v) * df['a_[x]']

#2A_[x]
for i in range(121):
  res = 1
  if i < 119: res += (df.loc[i, '2p[x]'] * v2 ** 2) * df.loc[i + 2, '2a_x']
  if i < 120: res += (df.loc[i, '1p[x]'] * v2)
  df.loc[i, '2a_[x]'] = res
df['2A_[x]'] = 1 - (1 - v2) * df['2a_[x]']

# Iax
df.loc[120, 'Ia_x'] = 1
for i in reversed(range(120)): df.loc[i, 'Ia_x'] = 1 + v * df.loc[i, 'p_x'] * (df.loc[i + 1, 'Ia_x'] + df.loc[i + 1, 'a_x'] )

# IAx
df.loc[120, 'IA_x'] = v * df.loc[120, 'q_x']
for i in reversed(range(120)): df.loc[i, 'IA_x'] = v * df.loc[i, 'q_x'] + v * df.loc[i, 'p_x'] * (df.loc[i + 1, 'IA_x'] + df.loc[i + 1, 'A_x'])

# Ia[x]
for i in range(121):
  res = 1
  if i < 119: res += df.loc[i, '2p[x]'] * v ** 2 * (df.loc[i + 2, 'Ia_x'] + 2 * df.loc[i + 2, 'a_x'])
  if i < 120: res += 2 * df.loc[i, '1p[x]'] * v
  df.loc[i, 'Ia_[x]'] = res

# IA[x]
for i in range(121):
  res = v * df.loc[i, '1q[x]'] 
  if i < 119: res += v ** 2 * df.loc[i, '2p[x]'] * (df.loc[i + 2, 'IA_x'] + 2 * df.loc[i + 2, 'A_x'])
  if i < 120: res += 2 * v ** 2 * df.loc[i, '1p[x]'] * df.loc[i, '1q[x+1]']
  df.loc[i, 'IA_[x]'] = res

df.drop(['x', 'p_x', 'q_x', '2a_x', '1p[x]', '1q[x]', '2p[x]', '1q[x+1]', '2a_[x]'], axis=1, inplace=True)

final_order = ['a_[x]', 'A_[x]', '2A_[x]', 'a_x', 'A_x', '2A_x', 'Ia_[x]', 'IA_[x]', 'Ia_x', 'IA_x']
df = df[final_order]
print(df)