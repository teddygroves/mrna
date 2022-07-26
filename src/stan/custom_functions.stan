/* This file is for your stan functions */

vector dPdt(real time, vector c, matrix S, vector p){
  vector[3] c_min_zero;
  for (i in 1:3){
    c_min_zero[i] = c[i] < 0 ? 0 : c[i];
  }
  return S * [p[1] * c_min_zero[1],
              p[2] * c_min_zero[2],
              p[3] * c_min_zero[2],
              p[5] * c_min_zero[3],
              p[5] * c_min_zero[2],
              p[5] * c_min_zero[1],
              p[4] * c_min_zero[3] * c_min_zero[1]]';
}
