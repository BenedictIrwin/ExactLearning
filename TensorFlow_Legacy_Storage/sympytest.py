import scipy.stats as st

class my_pdf(st.rv_continuous):
  def _pdf(self,x):
    return 3*x**2  # Normalized over its range, in this case [0,1]

my_cv = my_pdf(a=0, b=1, name='my_pdf')
my_cv.random()
