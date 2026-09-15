/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Numerical kernels for the pinker.kinematics backend.
 *
 * All functions operate on raw C double arrays passed as NumPy arrays from
 * the Python layer. Conventions follow Pinocchio:
 *
 * - Rotation matrices are 3x3, stored row-major as double[9].
 * - Quaternions are stored as (x, y, z, w), like in configuration vectors.
 * - Spatial (twist) vectors are (linear[3], angular[3]).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* Taylor series switching thresholds, as in Pinocchio: eps^(1/(degree+1)). */
static const double DBL_EPS = 2.220446049250313e-16;
#define PREC_2 6.0554544523933395e-06 /* eps^(1/3) */
#define PREC_3 1.2207031250000002e-04 /* eps^(1/4) */

/* ------------------------------------------------------------------ */
/* Small linear algebra helpers                                       */
/* ------------------------------------------------------------------ */

/* c = a x b (c must not alias a or b) */
static void cross3(double *c, const double *a, const double *b) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

/* a . b */
static double dot3(const double *a, const double *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/* y = R x for a 3x3 row-major matrix (y must not alias x) */
static void mat_vec(double *y, const double *R, const double *x) {
  y[0] = R[0] * x[0] + R[1] * x[1] + R[2] * x[2];
  y[1] = R[3] * x[0] + R[4] * x[1] + R[5] * x[2];
  y[2] = R[6] * x[0] + R[7] * x[1] + R[8] * x[2];
}

/* y = R^T x for a 3x3 row-major matrix (y must not alias x) */
static void mat_tvec(double *y, const double *R, const double *x) {
  y[0] = R[0] * x[0] + R[3] * x[1] + R[6] * x[2];
  y[1] = R[1] * x[0] + R[4] * x[1] + R[7] * x[2];
  y[2] = R[2] * x[0] + R[5] * x[1] + R[8] * x[2];
}

/* C = A B for 3x3 row-major matrices (C must not alias A or B) */
static void mat_mul(double *C, const double *A, const double *B) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C[3 * i + j] =
          A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    }
  }
}

/* 3x3 identity matrix */
static void mat_eye(double *R) {
  R[0] = 1.0;
  R[1] = 0.0;
  R[2] = 0.0;
  R[3] = 0.0;
  R[4] = 1.0;
  R[5] = 0.0;
  R[6] = 0.0;
  R[7] = 0.0;
  R[8] = 1.0;
}

/* ------------------------------------------------------------------ */
/* Quaternions                                                        */
/* ------------------------------------------------------------------ */

/*
 * Quaternions are represented as (x, y, z, w), vector part first, which is
 * how they are laid out in configuration vectors. For instance, the neutral
 * free-flyer configuration is (0, 0, 0, 0, 0, 0, 1), its trailing 1 being w.
 */

/* R = rotation matrix of the unit quaternion q */
static void quat_to_mat(double *R, const double *q) {
  double x = q[0], y = q[1], z = q[2], w = q[3];
  double x2 = x * x, y2 = y * y, z2 = z * z;
  double xy = x * y, xz = x * z, yz = y * z;
  double wx = w * x, wy = w * y, wz = w * z;
  R[0] = 1.0 - 2.0 * (y2 + z2);
  R[1] = 2.0 * (xy - wz);
  R[2] = 2.0 * (xz + wy);
  R[3] = 2.0 * (xy + wz);
  R[4] = 1.0 - 2.0 * (x2 + z2);
  R[5] = 2.0 * (yz - wx);
  R[6] = 2.0 * (xz - wy);
  R[7] = 2.0 * (yz + wx);
  R[8] = 1.0 - 2.0 * (x2 + y2);
}

/* out = a * b (Hamilton product) */
static void quat_mul(double *out, const double *a, const double *b) {
  double ax = a[0], ay = a[1], az = a[2], aw = a[3];
  double bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
}

/* out = conjugate of q, the inverse rotation for a unit quaternion */
static void quat_conj(double *out, const double *q) {
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = q[3];
}

/* out = R(q) v */
static void quat_rotate(double *out, const double *q, const double *v) {
  /* out = v + 2 w (u x v) + 2 (u x (u x v)) with q = (u, w) */
  double t[3], u[3] = {q[0], q[1], q[2]};
  cross3(t, u, v);
  t[0] *= 2.0;
  t[1] *= 2.0;
  t[2] *= 2.0;
  double s[3];
  cross3(s, u, t);
  out[0] = v[0] + q[3] * t[0] + s[0];
  out[1] = v[1] + q[3] * t[1] + s[1];
  out[2] = v[2] + q[3] * t[2] + s[2];
}

/* q <- q / |q|, leaving a zero quaternion untouched */
static void quat_normalize(double *q) {
  double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n > 0.0) {
    double inv = 1.0 / n;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
  }
}

/* R = c I + s [a]x + (1 - c) a a^T for a unit axis a */
static void rodrigues(double *R, const double *a, double c, double s) {
  double k = 1.0 - c;
  R[0] = c + k * a[0] * a[0];
  R[1] = k * a[0] * a[1] - s * a[2];
  R[2] = k * a[0] * a[2] + s * a[1];
  R[3] = k * a[1] * a[0] + s * a[2];
  R[4] = c + k * a[1] * a[1];
  R[5] = k * a[1] * a[2] - s * a[0];
  R[6] = k * a[2] * a[0] - s * a[1];
  R[7] = k * a[2] * a[1] + s * a[0];
  R[8] = c + k * a[2] * a[2];
}

/* ------------------------------------------------------------------ */
/* SO(3): exp, log, Jlog                                              */
/* ------------------------------------------------------------------ */

/* Quaternion of the rotation vector omega (exp3), q = (x, y, z, w). */
static void exp3_quat(double *q, const double *omega) {
  double t2 = dot3(omega, omega);
  double t = sqrt(t2);
  double s; /* sin(t/2) / t */
  if (t > PREC_3) {
    s = sin(0.5 * t) / t;
    q[3] = cos(0.5 * t);
  } else {
    s = 0.5 - t2 / 48.0;
    q[3] = 1.0 - t2 / 8.0 + t2 * t2 / 384.0;
  }
  q[0] = s * omega[0];
  q[1] = s * omega[1];
  q[2] = s * omega[2];
}

/* Rotation vector of a unit quaternion (x, y, z, w). */
static void quat_log3(double *w, double *theta_out, const double *q) {
  double s = (q[3] < 0.0) ? -1.0 : 1.0;
  double v[3] = {s * q[0], s * q[1], s * q[2]};
  double qw = s * q[3];
  double n = sqrt(dot3(v, v));
  double theta = 2.0 * atan2(n, qw);
  if (n > PREC_3) {
    double c = theta / n;
    w[0] = c * v[0];
    w[1] = c * v[1];
    w[2] = c * v[2];
  } else {
    /* theta / n -> 2 / qw * (1 - n^2 / (3 qw^2)) as n -> 0 */
    double c = 2.0 / qw * (1.0 - n * n / (3.0 * qw * qw));
    w[0] = c * v[0];
    w[1] = c * v[1];
    w[2] = c * v[2];
  }
  if (theta_out) *theta_out = theta;
}

/*
 * Rotation vector of a rotation matrix. Ported from Pinocchio log3_impl
 * (spatial/log.hxx), including the renormalization of R and the treatment
 * of the singular region around theta = pi.
 */
static void log3(double *w, double *theta_out, const double *R_in) {
  /* Renormalize: columns c0/|c0|, c1/|c1|, c2 = c0 x c1, c0 = c1 x c2. */
  double c0[3] = {R_in[0], R_in[3], R_in[6]};
  double c1[3] = {R_in[1], R_in[4], R_in[7]};
  double n0 = sqrt(dot3(c0, c0)), n1 = sqrt(dot3(c1, c1));
  for (int i = 0; i < 3; i++) {
    c0[i] /= n0;
    c1[i] /= n1;
  }
  double c2[3];
  cross3(c2, c0, c1);
  cross3(c0, c1, c2);
  double R[9] = {c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]};

  double tr = R[0] + R[4] + R[8];
  double cos_value = 0.5 * (tr - 1.0);

  /* antisymmetric part: skew(antisym) = (R - R^T) / 2 */
  double antisym[3] = {0.5 * (R[7] - R[5]), 0.5 * (R[2] - R[6]),
                       0.5 * (R[3] - R[1])};

  if (cos_value >= -1.0 + PREC_2) {
    /* Nominal case. */
    double theta;
    if (tr > 3.0 - PREC_2) {
      theta = sqrt(2.0 * (1.0 - cos_value) + DBL_EPS * DBL_EPS);
    } else {
      theta = acos(cos_value);
    }
    double a2 = dot3(antisym, antisym);
    double t;
    if (theta >= PREC_2) {
      t = theta / sin(theta);
    } else {
      t = 1.0 + a2 / 6.0 + a2 * a2 * 3.0 / 40.0;
    }
    w[0] = t * antisym[0];
    w[1] = t * antisym[1];
    w[2] = t * antisym[2];
    if (theta_out) *theta_out = theta;
    return;
  }

  /* Singular case, theta close to pi: pick the largest diagonal term. */
  double val[3];
  for (int i = 0; i < 3; i++) val[i] = 2.0 * R[4 * i] - tr + 1.0;
  int i0 = 0;
  if (val[1] > val[i0]) i0 = 1;
  if (val[2] > val[i0]) i0 = 2;
  int i1 = (i0 + 1) % 3, i2 = (i0 + 2) % 3;
  double sign = (R[3 * i2 + i1] >= R[3 * i1 + i2]) ? 1.0 : -1.0;
  double s = sqrt(val[i0] + DBL_EPS + DBL_EPS * DBL_EPS) * sign;
  double axis[3];
  axis[i0] = 0.5 * s;
  axis[i1] = (R[3 * i1 + i0] + R[3 * i0 + i1]) / (2.0 * s);
  axis[i2] = (R[3 * i2 + i0] + R[3 * i0 + i2]) / (2.0 * s);
  double qw = (R[3 * i2 + i1] - R[3 * i1 + i2]) / (2.0 * s);
  double axis_norm = sqrt(dot3(axis, axis));
  double theta = 2.0 * atan2(axis_norm, qw);
  for (int i = 0; i < 3; i++) w[i] = theta * axis[i] / axis_norm;
  if (theta_out) *theta_out = theta;
}

/* Jlog3 as a function of theta = |w| and w = log3(R). */
static void jlog3(double *J, double theta, const double *w) {
  double st = sin(theta), ct = cos(theta);
  double st_1mct = st / (1.0 - ct);
  double alpha, diag_value;
  if (theta < PREC_3) {
    alpha = 1.0 / 12.0 + theta * theta / 720.0;
    diag_value = 0.5 * (2.0 - theta * theta / 6.0);
  } else {
    alpha = 1.0 / (theta * theta) - st_1mct / (2.0 * theta);
    diag_value = 0.5 * (theta * st_1mct);
  }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) J[3 * i + j] = alpha * w[i] * w[j];
  J[0] += diag_value;
  J[4] += diag_value;
  J[8] += diag_value;
  /* += skew(w / 2) */
  J[1] += -0.5 * w[2];
  J[2] += 0.5 * w[1];
  J[3] += 0.5 * w[2];
  J[5] += -0.5 * w[0];
  J[6] += -0.5 * w[1];
  J[7] += 0.5 * w[0];
}

/* ------------------------------------------------------------------ */
/* SE(3): exp, log, Jlog                                              */
/* ------------------------------------------------------------------ */

/*
 * exp6 of the twist (v, w): rotation as quaternion, translation p = V(w) v
 * with V = I + (1 - cos t)/t^2 [w]x + (t - sin t)/t^3 [w]x^2.
 */
static void exp6_quat(double *q, double *p, const double *v, const double *w) {
  exp3_quat(q, w);
  double t2 = dot3(w, w);
  double a, b; /* a = (1 - cos t)/t^2, b = (t - sin t)/t^3 */
  if (t2 > PREC_3 * PREC_3) {
    double t = sqrt(t2);
    a = (1.0 - cos(t)) / t2;
    b = (t - sin(t)) / (t2 * t);
  } else {
    a = 0.5 - t2 / 24.0;
    b = 1.0 / 6.0 - t2 / 120.0;
  }
  double wxv[3], wxwxv[3];
  cross3(wxv, w, v);
  cross3(wxwxv, w, wxv);
  for (int i = 0; i < 3; i++) p[i] = v[i] + a * wxv[i] + b * wxwxv[i];
}

/*
 * log6 of the transform (R, p) into the twist out = (linear, angular).
 * Ported from Pinocchio log6_impl (matrix version).
 */
static void log6(double *out, const double *R, const double *p) {
  double w[3], theta;
  double antisym[3] = {0.5 * (R[7] - R[5]), 0.5 * (R[2] - R[6]),
                       0.5 * (R[3] - R[1])};
  double t2 = dot3(antisym, antisym);
  log3(w, &theta, R);

  double tr = R[0] + R[4] + R[8];
  double st = sin(theta), ct = cos(theta);
  double alpha, beta;
  if (tr >= 3.0 - PREC_2) {
    alpha = 1.0 - t2 / 12.0 - t2 * t2 / 720.0;
    beta = 1.0 / 12.0 + t2 / 720.0;
  } else {
    alpha = theta * st / (2.0 * (1.0 - ct));
    beta = 1.0 / (theta * theta) - st / (2.0 * theta * (1.0 - ct));
  }

  double wxp[3];
  cross3(wxp, w, p);
  double wdp = dot3(w, p);
  for (int i = 0; i < 3; i++)
    out[i] = alpha * p[i] - 0.5 * wxp[i] + beta * wdp * w[i];
  out[3] = w[0];
  out[4] = w[1];
  out[5] = w[2];
}

/*
 * Jlog6 of the transform (R, p), a 6x6 row-major matrix laid out as
 * [[A, B], [0, A]] with A = Jlog3. Ported from Pinocchio Jlog6_impl.
 */
static void jlog6(double *J, const double *R, const double *p) {
  double w[3], t;
  log3(w, &t, R);

  double A[9];
  jlog3(A, t, w);

  double t2 = t * t;
  double st = sin(t), ct = cos(t);
  double beta, beta_dot_over_theta;
  if (t < PREC_3) {
    beta = 1.0 / 12.0 + t2 / 720.0;
    beta_dot_over_theta = 1.0 / 360.0;
  } else {
    double tinv = 1.0 / t, t2inv = tinv * tinv;
    double inv_2_2ct = 1.0 / (2.0 * (1.0 - ct));
    beta = t2inv - st * tinv * inv_2_2ct;
    beta_dot_over_theta =
        -2.0 * t2inv * t2inv + (1.0 + st * tinv) * t2inv * inv_2_2ct;
  }

  double wTp = dot3(w, p);
  double v3_tmp[3];
  for (int i = 0; i < 3; i++)
    v3_tmp[i] = beta_dot_over_theta * wTp * w[i] -
                (t2 * beta_dot_over_theta + 2.0 * beta) * p[i];

  /* C = v3_tmp w^T + beta w p^T + wTp beta I + skew(p / 2) */
  double C[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      C[3 * i + j] = v3_tmp[i] * w[j] + beta * w[i] * p[j];
  C[0] += wTp * beta;
  C[4] += wTp * beta;
  C[8] += wTp * beta;
  C[1] += -0.5 * p[2];
  C[2] += 0.5 * p[1];
  C[3] += 0.5 * p[2];
  C[5] += -0.5 * p[0];
  C[6] += -0.5 * p[1];
  C[7] += 0.5 * p[0];

  double B[9];
  mat_mul(B, C, A);

  memset(J, 0, 36 * sizeof(double));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      J[6 * i + j] = A[3 * i + j];
      J[6 * i + j + 3] = B[3 * i + j];
      J[6 * (i + 3) + j + 3] = A[3 * i + j];
    }
  }
}

/* ------------------------------------------------------------------ */
/* SE(2) and SO(2), for planar and unbounded revolute joints          */
/* ------------------------------------------------------------------ */

/* Angle of a 2D rotation given by (c, s). Ported from Pinocchio SO2 log. */
static double so2_log(double c, double s) {
  double tr = 2.0 * c;
  if (tr > 2.0) return 0.0;
  if (tr < -2.0) return (s >= 0.0) ? M_PI : -M_PI;
  if (tr > 2.0 - 1e-2) return asin(s);
  return (s >= 0.0) ? acos(c) : -acos(c);
}

/* SE(2) exp of v = (vx, vy, omega): rotation (c, s), translation t[2]. */
static void se2_exp(double *c, double *s, double *t, const double *v) {
  double omega = v[2];
  double cv = cos(omega), sv = sin(omega);
  *c = cv;
  *s = sv;
  if (fabs(omega) > 1e-14) {
    /* t = (I - R) J v / omega with J = [[0, -1], [1, 0]] */
    double jx = -v[1], jy = v[0];
    t[0] = (jx - (cv * jx - sv * jy)) / omega;
    t[1] = (jy - (sv * jx + cv * jy)) / omega;
  } else {
    t[0] = v[0];
    t[1] = v[1];
  }
}

/* SE(2) log of the transform ((c, s), p): out = (vx, vy, theta). */
static void se2_log(double *out, double c, double s, const double *p) {
  double t = so2_log(c, s);
  double tabs = fabs(t);
  double t2 = t * t;
  double alpha;
  if (tabs < 1e-4) {
    alpha = 1.0 - t2 / 12.0 - t2 * t2 / 720.0;
  } else {
    double st = sin(tabs), ct = cos(tabs);
    alpha = tabs * st / (2.0 * (1.0 - ct));
  }
  out[0] = alpha * p[0] + 0.5 * t * p[1];
  out[1] = alpha * p[1] - 0.5 * t * p[0];
  out[2] = t;
}

/* SE(2) Jlog of the transform ((c, s), p): 3x3 row-major. */
static void se2_jlog(double *J, double c, double s, const double *p) {
  double t = so2_log(c, s);
  double tabs = fabs(t);
  double t2 = t * t;
  double alpha, alpha_dot;
  if (tabs < 1e-4) {
    alpha = 1.0 - t2 / 12.0;
    alpha_dot = -t / 6.0 - t2 * t / 180.0;
  } else {
    double st = sin(t), ct = cos(t);
    double inv_2_1_ct = 0.5 / (1.0 - ct);
    alpha = t * st * inv_2_1_ct;
    alpha_dot = (st - t) * inv_2_1_ct;
  }
  /* topLeft = V R with V = [[alpha, t/2], [-t/2, alpha]], R = [[c,-s],[s,c]] */
  J[0] = alpha * c + 0.5 * t * s;
  J[1] = -alpha * s + 0.5 * t * c;
  J[3] = -0.5 * t * c + alpha * s;
  J[4] = 0.5 * t * s + alpha * c;
  J[2] = alpha_dot * p[0] + 0.5 * p[1];
  J[5] = -0.5 * p[0] + alpha_dot * p[1];
  J[6] = 0.0;
  J[7] = 0.0;
  J[8] = 1.0;
}

/* ------------------------------------------------------------------ */
/* Joint models                                                       */
/* ------------------------------------------------------------------ */

enum {
  JT_UNIVERSE = 0,
  JT_REVOLUTE = 1,
  JT_PRISMATIC = 2,
  JT_REVOLUTE_UNBOUNDED = 3,
  JT_SPHERICAL = 4,
  JT_PLANAR = 5,
  JT_FREEFLYER = 6
};

static const int JOINT_NQ[] = {0, 1, 1, 2, 4, 4, 7};
static const int JOINT_NV[] = {0, 1, 1, 1, 3, 3, 6};

/* Local transform (R, p) of a joint for its configuration segment qj. */
static void joint_transform(int jt, const double *axis, const double *qj,
                            double *R, double *p) {
  p[0] = 0.0;
  p[1] = 0.0;
  p[2] = 0.0;
  switch (jt) {
    case JT_REVOLUTE:
      rodrigues(R, axis, cos(qj[0]), sin(qj[0]));
      break;
    case JT_PRISMATIC:
      mat_eye(R);
      p[0] = axis[0] * qj[0];
      p[1] = axis[1] * qj[0];
      p[2] = axis[2] * qj[0];
      break;
    case JT_REVOLUTE_UNBOUNDED:
      rodrigues(R, axis, qj[0], qj[1]);
      break;
    case JT_SPHERICAL:
      quat_to_mat(R, qj);
      break;
    case JT_PLANAR:
      mat_eye(R);
      R[0] = qj[2];
      R[1] = -qj[3];
      R[3] = qj[3];
      R[4] = qj[2];
      p[0] = qj[0];
      p[1] = qj[1];
      break;
    case JT_FREEFLYER:
      quat_to_mat(R, qj + 3);
      p[0] = qj[0];
      p[1] = qj[1];
      p[2] = qj[2];
      break;
    default:
      mat_eye(R);
      break;
  }
}

/*
 * Fill the world-frame Jacobian columns of one joint into J (6 x nv,
 * row-major). (R, p) is the world placement of the joint. Column layout is
 * (linear at world origin, angular), like Pinocchio's data.J.
 */
static void joint_jacobian_cols(int jt, const double *axis, const double *R,
                                const double *p, int idx_v, double *J, int nv) {
  double wcol[3], vcol[3];
  switch (jt) {
    case JT_REVOLUTE:
    case JT_REVOLUTE_UNBOUNDED:
      mat_vec(wcol, R, axis);
      cross3(vcol, p, wcol);
      for (int r = 0; r < 3; r++) {
        J[r * nv + idx_v] = vcol[r];
        J[(r + 3) * nv + idx_v] = wcol[r];
      }
      break;
    case JT_PRISMATIC:
      mat_vec(vcol, R, axis);
      for (int r = 0; r < 3; r++) {
        J[r * nv + idx_v] = vcol[r];
        J[(r + 3) * nv + idx_v] = 0.0;
      }
      break;
    case JT_SPHERICAL:
      for (int i = 0; i < 3; i++) {
        wcol[0] = R[i];
        wcol[1] = R[3 + i];
        wcol[2] = R[6 + i];
        cross3(vcol, p, wcol);
        for (int r = 0; r < 3; r++) {
          J[r * nv + idx_v + i] = vcol[r];
          J[(r + 3) * nv + idx_v + i] = wcol[r];
        }
      }
      break;
    case JT_PLANAR:
      for (int i = 0; i < 2; i++) {
        for (int r = 0; r < 3; r++) {
          J[r * nv + idx_v + i] = R[3 * r + i];
          J[(r + 3) * nv + idx_v + i] = 0.0;
        }
      }
      wcol[0] = R[2];
      wcol[1] = R[5];
      wcol[2] = R[8];
      cross3(vcol, p, wcol);
      for (int r = 0; r < 3; r++) {
        J[r * nv + idx_v + 2] = vcol[r];
        J[(r + 3) * nv + idx_v + 2] = wcol[r];
      }
      break;
    case JT_FREEFLYER:
      for (int i = 0; i < 3; i++) {
        for (int r = 0; r < 3; r++) {
          J[r * nv + idx_v + i] = R[3 * r + i];
          J[(r + 3) * nv + idx_v + i] = 0.0;
        }
      }
      for (int i = 0; i < 3; i++) {
        wcol[0] = R[i];
        wcol[1] = R[3 + i];
        wcol[2] = R[6 + i];
        cross3(vcol, p, wcol);
        for (int r = 0; r < 3; r++) {
          J[r * nv + idx_v + 3 + i] = vcol[r];
          J[(r + 3) * nv + idx_v + 3 + i] = wcol[r];
        }
      }
      break;
    default:
      break;
  }
}

/* ------------------------------------------------------------------ */
/* NumPy argument helpers                                             */
/* ------------------------------------------------------------------ */

/*
 * Data pointer of a C-contiguous NumPy array of the expected type, checking
 * its size when size >= 0. Returns NULL with an exception set on mismatch.
 */
static void *check_array(PyObject *o, int typenum, npy_intp size,
                         const char *name) {
  if (!PyArray_Check(o)) {
    PyErr_Format(PyExc_TypeError, "%s: expected a NumPy array", name);
    return NULL;
  }
  PyArrayObject *a = (PyArrayObject *)o;
  if (PyArray_TYPE(a) != typenum || !PyArray_IS_C_CONTIGUOUS(a)) {
    PyErr_Format(PyExc_TypeError, "%s: expected a C-contiguous array of %s",
                 name, typenum == NPY_FLOAT64 ? "float64" : "int32");
    return NULL;
  }
  if (size >= 0 && PyArray_SIZE(a) != size) {
    PyErr_Format(PyExc_ValueError, "%s: expected %ld elements, got %ld", name,
                 (long)size, (long)PyArray_SIZE(a));
    return NULL;
  }
  return PyArray_DATA(a);
}

/* Data pointer of a float64 array of `size` elements (-1: any size) */
static double *as_f64(PyObject *o, npy_intp size, const char *name) {
  return (double *)check_array(o, NPY_FLOAT64, size, name);
}

/* Data pointer of an int32 array of `size` elements (-1: any size) */
static int32_t *as_i32(PyObject *o, npy_intp size, const char *name) {
  return (int32_t *)check_array(o, NPY_INT32, size, name);
}

/* Fresh uninitialized float64 array of the given shape */
static PyObject *new_f64(int ndim, const npy_intp *dims) {
  return PyArray_SimpleNew(ndim, (npy_intp *)dims, NPY_FLOAT64);
}

/* ------------------------------------------------------------------ */
/* Kinematics kernels                                                 */
/* ------------------------------------------------------------------ */

/*
 * forward_kinematics(jtype, parent, idx_q, axis, jp_rot, jp_trans, q,
 *                    oMi_rot, oMi_trans)
 *
 * Compute world placements of all joints. Arrays are indexed by joint id,
 * with joint 0 being the universe.
 */
static PyObject *py_forward_kinematics(PyObject *self, PyObject *args) {
  PyObject *o_jtype, *o_parent, *o_idx_q, *o_axis, *o_jp_rot, *o_jp_trans, *o_q,
      *o_rot, *o_trans;
  if (!PyArg_ParseTuple(args, "OOOOOOOOO", &o_jtype, &o_parent, &o_idx_q,
                        &o_axis, &o_jp_rot, &o_jp_trans, &o_q, &o_rot,
                        &o_trans))
    return NULL;

  int32_t *jtype = as_i32(o_jtype, -1, "jtype");
  if (!jtype) return NULL;
  npy_intp nj = PyArray_SIZE((PyArrayObject *)o_jtype);
  int32_t *parent = as_i32(o_parent, nj, "parent");
  int32_t *idx_q = as_i32(o_idx_q, nj, "idx_q");
  double *axis = as_f64(o_axis, 3 * nj, "axis");
  double *jp_rot = as_f64(o_jp_rot, 9 * nj, "jp_rot");
  double *jp_trans = as_f64(o_jp_trans, 3 * nj, "jp_trans");
  double *q = as_f64(o_q, -1, "q");
  double *oMi_rot = as_f64(o_rot, 9 * nj, "oMi_rot");
  double *oMi_trans = as_f64(o_trans, 3 * nj, "oMi_trans");
  if (!parent || !idx_q || !axis || !jp_rot || !jp_trans || !q || !oMi_rot ||
      !oMi_trans)
    return NULL;

  mat_eye(oMi_rot);
  oMi_trans[0] = 0.0;
  oMi_trans[1] = 0.0;
  oMi_trans[2] = 0.0;

  for (npy_intp j = 1; j < nj; j++) {
    double Rj[9], pj[3];
    joint_transform(jtype[j], axis + 3 * j, q + idx_q[j], Rj, pj);

    /* Placement in parent joint frame: jp * (Rj, pj) */
    double R_pc[9], p_pc[3], tmp[3];
    mat_mul(R_pc, jp_rot + 9 * j, Rj);
    mat_vec(tmp, jp_rot + 9 * j, pj);
    for (int i = 0; i < 3; i++) p_pc[i] = jp_trans[3 * j + i] + tmp[i];

    /* World placement: oMi[parent] * (R_pc, p_pc) */
    const double *Rp = oMi_rot + 9 * parent[j];
    const double *pp = oMi_trans + 3 * parent[j];
    mat_mul(oMi_rot + 9 * j, Rp, R_pc);
    mat_vec(tmp, Rp, p_pc);
    for (int i = 0; i < 3; i++) oMi_trans[3 * j + i] = pp[i] + tmp[i];
  }
  Py_RETURN_NONE;
}

/*
 * joint_jacobians(jtype, idx_v, axis, oMi_rot, oMi_trans, J)
 *
 * Fill the full model Jacobian J (6 x nv, row-major) with world-frame
 * columns, like Pinocchio's computeJointJacobians fills data.J.
 */
static PyObject *py_joint_jacobians(PyObject *self, PyObject *args) {
  PyObject *o_jtype, *o_idx_v, *o_axis, *o_rot, *o_trans, *o_J;
  int nv;
  if (!PyArg_ParseTuple(args, "OOOOOOi", &o_jtype, &o_idx_v, &o_axis, &o_rot,
                        &o_trans, &o_J, &nv))
    return NULL;

  int32_t *jtype = as_i32(o_jtype, -1, "jtype");
  if (!jtype) return NULL;
  npy_intp nj = PyArray_SIZE((PyArrayObject *)o_jtype);
  int32_t *idx_v = as_i32(o_idx_v, nj, "idx_v");
  double *axis = as_f64(o_axis, 3 * nj, "axis");
  double *oMi_rot = as_f64(o_rot, 9 * nj, "oMi_rot");
  double *oMi_trans = as_f64(o_trans, 3 * nj, "oMi_trans");
  double *J = as_f64(o_J, 6 * (npy_intp)nv, "J");
  if (!idx_v || !axis || !oMi_rot || !oMi_trans || !J) return NULL;

  for (npy_intp j = 1; j < nj; j++)
    joint_jacobian_cols(jtype[j], axis + 3 * j, oMi_rot + 9 * j,
                        oMi_trans + 3 * j, idx_v[j], J, nv);
  Py_RETURN_NONE;
}

/*
 * frame_placements(fparent, fp_rot, fp_trans, oMi_rot, oMi_trans,
 *                  oMf_rot, oMf_trans)
 *
 * Compose each frame placement in its parent joint with that joint's world
 * placement, giving the world placement of every frame.
 */
static PyObject *py_frame_placements(PyObject *self, PyObject *args) {
  PyObject *o_fparent, *o_fp_rot, *o_fp_trans, *o_rot, *o_trans, *o_frot,
      *o_ftrans;
  if (!PyArg_ParseTuple(args, "OOOOOOO", &o_fparent, &o_fp_rot, &o_fp_trans,
                        &o_rot, &o_trans, &o_frot, &o_ftrans))
    return NULL;

  int32_t *fparent = as_i32(o_fparent, -1, "fparent");
  if (!fparent) return NULL;
  npy_intp nf = PyArray_SIZE((PyArrayObject *)o_fparent);
  double *fp_rot = as_f64(o_fp_rot, 9 * nf, "fp_rot");
  double *fp_trans = as_f64(o_fp_trans, 3 * nf, "fp_trans");
  double *oMi_rot = as_f64(o_rot, -1, "oMi_rot");
  double *oMi_trans = as_f64(o_trans, -1, "oMi_trans");
  double *oMf_rot = as_f64(o_frot, 9 * nf, "oMf_rot");
  double *oMf_trans = as_f64(o_ftrans, 3 * nf, "oMf_trans");
  if (!fp_rot || !fp_trans || !oMi_rot || !oMi_trans || !oMf_rot || !oMf_trans)
    return NULL;

  for (npy_intp f = 0; f < nf; f++) {
    const double *Rp = oMi_rot + 9 * fparent[f];
    const double *pp = oMi_trans + 3 * fparent[f];
    double tmp[3];
    mat_mul(oMf_rot + 9 * f, Rp, fp_rot + 9 * f);
    mat_vec(tmp, Rp, fp_trans + 3 * f);
    for (int i = 0; i < 3; i++) oMf_trans[3 * f + i] = pp[i] + tmp[i];
  }
  Py_RETURN_NONE;
}

/* Reference frames (values should match those in model.py). */
enum { RF_WORLD = 0, RF_LOCAL = 1, RF_LOCAL_WORLD_ALIGNED = 2 };

/*
 * frame_jacobian(J, R_f, p_f, support, rf, out)
 *
 * Extract the Jacobian of a frame with world placement (R_f, p_f) from the
 * full model Jacobian J (6 x nv). `support` lists the tangent-space columns of
 * the joints supporting the frame; other columns are left at zero.
 */
static PyObject *py_frame_jacobian(PyObject *self, PyObject *args) {
  PyObject *o_J, *o_Rf, *o_pf, *o_support, *o_out;
  int rf, nv;
  if (!PyArg_ParseTuple(args, "OOOOiOi", &o_J, &o_Rf, &o_pf, &o_support, &rf,
                        &o_out, &nv))
    return NULL;

  double *J = as_f64(o_J, 6 * (npy_intp)nv, "J");
  double *Rf = as_f64(o_Rf, 9, "R_f");
  double *pf = as_f64(o_pf, 3, "p_f");
  int32_t *support = as_i32(o_support, -1, "support");
  double *out = as_f64(o_out, 6 * (npy_intp)nv, "out");
  if (!J || !Rf || !pf || !support || !out) return NULL;
  npy_intp ns = PyArray_SIZE((PyArrayObject *)o_support);

  memset(out, 0, 6 * (size_t)nv * sizeof(double));
  for (npy_intp k = 0; k < ns; k++) {
    int c = support[k];
    double v0[3], w[3];
    for (int r = 0; r < 3; r++) {
      v0[r] = J[r * nv + c];
      w[r] = J[(r + 3) * nv + c];
    }
    double vf[3], wxpf[3];
    cross3(wxpf, w, pf);
    for (int r = 0; r < 3; r++) vf[r] = v0[r] + wxpf[r];
    switch (rf) {
      case RF_WORLD:
        for (int r = 0; r < 3; r++) {
          out[r * nv + c] = v0[r];
          out[(r + 3) * nv + c] = w[r];
        }
        break;
      case RF_LOCAL: {
        double vl[3], wl[3];
        mat_tvec(vl, Rf, vf);
        mat_tvec(wl, Rf, w);
        for (int r = 0; r < 3; r++) {
          out[r * nv + c] = vl[r];
          out[(r + 3) * nv + c] = wl[r];
        }
        break;
      }
      case RF_LOCAL_WORLD_ALIGNED:
        for (int r = 0; r < 3; r++) {
          out[r * nv + c] = vf[r];
          out[(r + 3) * nv + c] = w[r];
        }
        break;
      default:
        PyErr_SetString(PyExc_ValueError, "unknown reference frame");
        return NULL;
    }
  }
  Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Lie-group operations on configuration vectors                      */
/* ------------------------------------------------------------------ */

/*
 * integrate(jtype, idx_q, idx_v, q, v, qout)
 *
 * qout_j = q_j * exp(v_j) for each joint, each on its own Lie group.
 */
static PyObject *py_integrate(PyObject *self, PyObject *args) {
  PyObject *o_jtype, *o_idx_q, *o_idx_v, *o_q, *o_v, *o_qout;
  if (!PyArg_ParseTuple(args, "OOOOOO", &o_jtype, &o_idx_q, &o_idx_v, &o_q,
                        &o_v, &o_qout))
    return NULL;

  int32_t *jtype = as_i32(o_jtype, -1, "jtype");
  if (!jtype) return NULL;
  npy_intp nj = PyArray_SIZE((PyArrayObject *)o_jtype);
  int32_t *idx_q = as_i32(o_idx_q, nj, "idx_q");
  int32_t *idx_v = as_i32(o_idx_v, nj, "idx_v");
  double *q = as_f64(o_q, -1, "q");
  double *v = as_f64(o_v, -1, "v");
  double *qout = as_f64(o_qout, PyArray_SIZE((PyArrayObject *)o_q), "qout");
  if (!idx_q || !idx_v || !q || !v || !qout) return NULL;

  for (npy_intp j = 1; j < nj; j++) {
    const double *qj = q + idx_q[j];
    const double *vj = v + idx_v[j];
    double *oj = qout + idx_q[j];
    switch (jtype[j]) {
      case JT_REVOLUTE:
      case JT_PRISMATIC:
        oj[0] = qj[0] + vj[0];
        break;
      case JT_REVOLUTE_UNBOUNDED: {
        double co = cos(vj[0]), so = sin(vj[0]);
        oj[0] = co * qj[0] - so * qj[1];
        oj[1] = so * qj[0] + co * qj[1];
        break;
      }
      case JT_SPHERICAL: {
        double eq[4], res[4];
        exp3_quat(eq, vj);
        quat_mul(res, qj, eq);
        double dp =
            res[0] * qj[0] + res[1] * qj[1] + res[2] * qj[2] + res[3] * qj[3];
        if (dp < 0.0)
          for (int i = 0; i < 4; i++) res[i] = -res[i];
        quat_normalize(res);
        for (int i = 0; i < 4; i++) oj[i] = res[i];
        break;
      }
      case JT_PLANAR: {
        double c, s, t[2];
        se2_exp(&c, &s, t, vj);
        /* out = (R0 t + t0, R0 R e_0) */
        double c0 = qj[2], s0 = qj[3];
        oj[0] = qj[0] + c0 * t[0] - s0 * t[1];
        oj[1] = qj[1] + s0 * t[0] + c0 * t[1];
        oj[2] = c0 * c - s0 * s;
        oj[3] = s0 * c + c0 * s;
        break;
      }
      case JT_FREEFLYER: {
        double eq[4], ep[3], res[4], tmp[3];
        exp6_quat(eq, ep, vj, vj + 3);
        quat_rotate(tmp, qj + 3, ep);
        oj[0] = qj[0] + tmp[0];
        oj[1] = qj[1] + tmp[1];
        oj[2] = qj[2] + tmp[2];
        quat_mul(res, qj + 3, eq);
        double dp =
            res[0] * qj[3] + res[1] * qj[4] + res[2] * qj[5] + res[3] * qj[6];
        if (dp < 0.0)
          for (int i = 0; i < 4; i++) res[i] = -res[i];
        quat_normalize(res);
        for (int i = 0; i < 4; i++) oj[3 + i] = res[i];
        break;
      }
      default:
        break;
    }
  }
  Py_RETURN_NONE;
}

/*
 * difference(jtype, idx_q, idx_v, q0, q1, dout)
 *
 * dout[idx_v[j]:] = log(q0_j^-1 * q1_j) for each joint.
 */
static PyObject *py_difference(PyObject *self, PyObject *args) {
  PyObject *o_jtype, *o_idx_q, *o_idx_v, *o_q0, *o_q1, *o_d;
  if (!PyArg_ParseTuple(args, "OOOOOO", &o_jtype, &o_idx_q, &o_idx_v, &o_q0,
                        &o_q1, &o_d))
    return NULL;

  int32_t *jtype = as_i32(o_jtype, -1, "jtype");
  if (!jtype) return NULL;
  npy_intp nj = PyArray_SIZE((PyArrayObject *)o_jtype);
  int32_t *idx_q = as_i32(o_idx_q, nj, "idx_q");
  int32_t *idx_v = as_i32(o_idx_v, nj, "idx_v");
  double *q0 = as_f64(o_q0, -1, "q0");
  double *q1 = as_f64(o_q1, PyArray_SIZE((PyArrayObject *)o_q0), "q1");
  double *d = as_f64(o_d, -1, "dout");
  if (!idx_q || !idx_v || !q0 || !q1 || !d) return NULL;

  for (npy_intp j = 1; j < nj; j++) {
    const double *a = q0 + idx_q[j];
    const double *b = q1 + idx_q[j];
    double *dj = d + idx_v[j];
    switch (jtype[j]) {
      case JT_REVOLUTE:
      case JT_PRISMATIC:
        dj[0] = b[0] - a[0];
        break;
      case JT_REVOLUTE_UNBOUNDED: {
        /* Relative rotation: c = a . b, s = a x b */
        double c = a[0] * b[0] + a[1] * b[1];
        double s = a[0] * b[1] - a[1] * b[0];
        dj[0] = so2_log(c, s);
        break;
      }
      case JT_SPHERICAL: {
        double ainv[4], rel[4];
        quat_conj(ainv, a);
        quat_mul(rel, ainv, b);
        quat_log3(dj, NULL, rel);
        break;
      }
      case JT_PLANAR: {
        double c = a[2] * b[2] + a[3] * b[3];
        double s = a[2] * b[3] - a[3] * b[2];
        double dx = b[0] - a[0], dy = b[1] - a[1];
        double p[2] = {a[2] * dx + a[3] * dy, -a[3] * dx + a[2] * dy};
        se2_log(dj, c, s, p);
        break;
      }
      case JT_FREEFLYER: {
        double ainv[4], rel[4], dv_pre[3], dv[3], R[9];
        quat_conj(ainv, a + 3);
        quat_mul(rel, ainv, b + 3);
        for (int i = 0; i < 3; i++) dv_pre[i] = b[i] - a[i];
        quat_rotate(dv, ainv, dv_pre);
        quat_to_mat(R, rel);
        log6(dj, R, dv);
        break;
      }
      default:
        break;
    }
  }
  Py_RETURN_NONE;
}

/*
 * d_difference(jtype, idx_q, idx_v, q0, q1, arg, Jout, nv)
 *
 * Jacobian of difference(q0, q1) with respect to q0 (arg = 0) or q1
 * (arg = 1). Jout is nv x nv row-major, zeroed here, block-diagonal.
 */
static PyObject *py_d_difference(PyObject *self, PyObject *args) {
  PyObject *o_jtype, *o_idx_q, *o_idx_v, *o_q0, *o_q1, *o_J;
  int arg, nv;
  if (!PyArg_ParseTuple(args, "OOOOOiOi", &o_jtype, &o_idx_q, &o_idx_v, &o_q0,
                        &o_q1, &arg, &o_J, &nv))
    return NULL;

  int32_t *jtype = as_i32(o_jtype, -1, "jtype");
  if (!jtype) return NULL;
  npy_intp nj = PyArray_SIZE((PyArrayObject *)o_jtype);
  int32_t *idx_q = as_i32(o_idx_q, nj, "idx_q");
  int32_t *idx_v = as_i32(o_idx_v, nj, "idx_v");
  double *q0 = as_f64(o_q0, -1, "q0");
  double *q1 = as_f64(o_q1, PyArray_SIZE((PyArrayObject *)o_q0), "q1");
  double *J = as_f64(o_J, (npy_intp)nv * nv, "Jout");
  if (!idx_q || !idx_v || !q0 || !q1 || !J) return NULL;

  memset(J, 0, (size_t)nv * nv * sizeof(double));

#define JBLOCK(r, c) J[(npy_intp)(idx_v[j] + (r)) * nv + (idx_v[j] + (c))]

  for (npy_intp j = 1; j < nj; j++) {
    const double *a = q0 + idx_q[j];
    const double *b = q1 + idx_q[j];
    switch (jtype[j]) {
      case JT_REVOLUTE:
      case JT_PRISMATIC:
      case JT_REVOLUTE_UNBOUNDED:
        JBLOCK(0, 0) = (arg == 0) ? -1.0 : 1.0;
        break;
      case JT_SPHERICAL: {
        double ainv[4], rel[4], R[9], w[3], theta, Jl[9];
        quat_conj(ainv, a);
        quat_mul(rel, ainv, b);
        quat_to_mat(R, rel);
        log3(w, &theta, R);
        jlog3(Jl, theta, w);
        if (arg == 1) {
          for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) JBLOCK(r, c) = Jl[3 * r + c];
        } else {
          /* -Jlog3(R) R^T */
          for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
              double acc = 0.0;
              for (int k = 0; k < 3; k++) acc += Jl[3 * r + k] * R[3 * c + k];
              JBLOCK(r, c) = -acc;
            }
        }
        break;
      }
      case JT_PLANAR: {
        double c = a[2] * b[2] + a[3] * b[3];
        double s = a[2] * b[3] - a[3] * b[2];
        double dx = b[0] - a[0], dy = b[1] - a[1];
        double p[2] = {a[2] * dx + a[3] * dy, -a[3] * dx + a[2] * dy};
        double J1[9];
        se2_jlog(J1, c, s, p);
        if (arg == 1) {
          for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++) JBLOCK(r, cc) = J1[3 * r + cc];
        } else {
          /*
           * J0 = [[-R^T, R1^T pcross], [0, -1]] then J = J1 J0,
           * with pcross = (y1 - y0, -(x1 - x0)).
           */
          double J0[9];
          J0[0] = -c;
          J0[1] = -s;
          J0[3] = s;
          J0[4] = -c;
          /* R1^T pcross with R1 from (b[2], b[3]) */
          double px = dy, py = -dx;
          J0[2] = b[2] * px + b[3] * py;
          J0[5] = -b[3] * px + b[2] * py;
          J0[6] = 0.0;
          J0[7] = 0.0;
          J0[8] = -1.0;
          double Jf[9];
          mat_mul(Jf, J1, J0);
          for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++) JBLOCK(r, cc) = Jf[3 * r + cc];
        }
        break;
      }
      case JT_FREEFLYER: {
        double ainv[4], rel[4], dv_pre[3], t[3], R[9], J1[36];
        quat_conj(ainv, a + 3);
        quat_mul(rel, ainv, b + 3);
        for (int i = 0; i < 3; i++) dv_pre[i] = b[i] - a[i];
        quat_rotate(t, ainv, dv_pre);
        quat_to_mat(R, rel);
        jlog6(J1, R, t);
        if (arg == 1) {
          for (int r = 0; r < 6; r++)
            for (int c = 0; c < 6; c++) JBLOCK(r, c) = J1[6 * r + c];
        } else {
          /*
           * J0 = [[-R^T, skew(p1_p0) R^T], [0, -R^T]] with
           * p1_p0 = R1^T (t1 - t0), then J = J1 J0.
           */
          double binv[4], p1_p0[3], J0[36], Jf[36];
          quat_conj(binv, b + 3);
          quat_rotate(p1_p0, binv, dv_pre);
          memset(J0, 0, sizeof(J0));
          double sk[9] = {0.0,       -p1_p0[2], p1_p0[1], p1_p0[2], 0.0,
                          -p1_p0[0], -p1_p0[1], p1_p0[0], 0.0};
          double skRt[9];
          /* skRt = sk * R^T */
          for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
              double acc = 0.0;
              for (int k = 0; k < 3; k++) acc += sk[3 * r + k] * R[3 * c + k];
              skRt[3 * r + c] = acc;
            }
          for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
              J0[6 * r + c] = -R[3 * c + r];
              J0[6 * r + c + 3] = skRt[3 * r + c];
              J0[6 * (r + 3) + c + 3] = -R[3 * c + r];
            }
          for (int r = 0; r < 6; r++)
            for (int c = 0; c < 6; c++) {
              double acc = 0.0;
              for (int k = 0; k < 6; k++) acc += J1[6 * r + k] * J0[6 * k + c];
              Jf[6 * r + c] = acc;
            }
          for (int r = 0; r < 6; r++)
            for (int c = 0; c < 6; c++) JBLOCK(r, c) = Jf[6 * r + c];
        }
        break;
      }
      default:
        break;
    }
  }
#undef JBLOCK
  Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Center of mass                                                     */
/* ------------------------------------------------------------------ */

/*
 * center_of_mass(parent, mass, lever, oMi_rot, oMi_trans, com_out)
 *
 * Whole-robot center of mass in the world frame. As in Pinocchio, the
 * inertia attached to the universe (joint 0) is not included.
 * Returns the total mass.
 */
static PyObject *py_center_of_mass(PyObject *self, PyObject *args) {
  PyObject *o_parent, *o_mass, *o_lever, *o_rot, *o_trans, *o_com;
  if (!PyArg_ParseTuple(args, "OOOOOO", &o_parent, &o_mass, &o_lever, &o_rot,
                        &o_trans, &o_com))
    return NULL;

  int32_t *parent = as_i32(o_parent, -1, "parent");
  if (!parent) return NULL;
  npy_intp nj = PyArray_SIZE((PyArrayObject *)o_parent);
  double *mass = as_f64(o_mass, nj, "mass");
  double *lever = as_f64(o_lever, 3 * nj, "lever");
  double *oMi_rot = as_f64(o_rot, 9 * nj, "oMi_rot");
  double *oMi_trans = as_f64(o_trans, 3 * nj, "oMi_trans");
  double *com = as_f64(o_com, 3, "com_out");
  if (!mass || !lever || !oMi_rot || !oMi_trans || !com) return NULL;

  double total_mass = 0.0;
  com[0] = 0.0;
  com[1] = 0.0;
  com[2] = 0.0;
  for (npy_intp j = 1; j < nj; j++) {
    double x[3];
    mat_vec(x, oMi_rot + 9 * j, lever + 3 * j);
    for (int i = 0; i < 3; i++)
      com[i] += mass[j] * (x[i] + oMi_trans[3 * j + i]);
    total_mass += mass[j];
  }
  if (total_mass > 0.0)
    for (int i = 0; i < 3; i++) com[i] /= total_mass;
  return PyFloat_FromDouble(total_mass);
}

/*
 * com_jacobian(jtype, parent, idx_v, mass, lever, oMi_rot, oMi_trans,
 *              J, Jcom, com_out, nv)
 *
 * Center-of-mass Jacobian (3 x nv). J is the full model Jacobian in world
 * frame (as computed by joint_jacobians). Also fills com_out.
 */
static PyObject *py_com_jacobian(PyObject *self, PyObject *args) {
  PyObject *o_jtype, *o_parent, *o_idx_v, *o_mass, *o_lever, *o_rot, *o_trans,
      *o_J, *o_Jcom, *o_com;
  int nv;
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOi", &o_jtype, &o_parent, &o_idx_v,
                        &o_mass, &o_lever, &o_rot, &o_trans, &o_J, &o_Jcom,
                        &o_com, &nv))
    return NULL;

  int32_t *jtype = as_i32(o_jtype, -1, "jtype");
  if (!jtype) return NULL;
  npy_intp nj = PyArray_SIZE((PyArrayObject *)o_jtype);
  int32_t *parent = as_i32(o_parent, nj, "parent");
  int32_t *idx_v = as_i32(o_idx_v, nj, "idx_v");
  double *mass = as_f64(o_mass, nj, "mass");
  double *lever = as_f64(o_lever, 3 * nj, "lever");
  double *oMi_rot = as_f64(o_rot, 9 * nj, "oMi_rot");
  double *oMi_trans = as_f64(o_trans, 3 * nj, "oMi_trans");
  double *J = as_f64(o_J, 6 * (npy_intp)nv, "J");
  double *Jcom = as_f64(o_Jcom, 3 * (npy_intp)nv, "Jcom");
  double *com = as_f64(o_com, 3, "com_out");
  if (!parent || !idx_v || !mass || !lever || !oMi_rot || !oMi_trans || !J ||
      !Jcom || !com)
    return NULL;

  /* Subtree mass and mass-weighted center, in world coordinates. */
  double *sub_mass = (double *)calloc((size_t)nj, sizeof(double));
  double *sub_mcom = (double *)calloc((size_t)nj * 3, sizeof(double));
  if (!sub_mass || !sub_mcom) {
    free(sub_mass);
    free(sub_mcom);
    return PyErr_NoMemory();
  }
  for (npy_intp j = 1; j < nj; j++) {
    double x[3];
    mat_vec(x, oMi_rot + 9 * j, lever + 3 * j);
    sub_mass[j] = mass[j];
    for (int i = 0; i < 3; i++)
      sub_mcom[3 * j + i] = mass[j] * (x[i] + oMi_trans[3 * j + i]);
  }
  for (npy_intp j = nj - 1; j >= 1; j--) {
    sub_mass[parent[j]] += sub_mass[j];
    for (int i = 0; i < 3; i++)
      sub_mcom[3 * parent[j] + i] += sub_mcom[3 * j + i];
  }
  double total_mass = sub_mass[0];

  memset(Jcom, 0, 3 * (size_t)nv * sizeof(double));
  for (npy_intp j = 1; j < nj; j++) {
    int nvj = JOINT_NV[jtype[j]];
    for (int k = 0; k < nvj; k++) {
      int c = idx_v[j] + k;
      double v0[3], w[3], wxm[3];
      for (int r = 0; r < 3; r++) {
        v0[r] = J[r * nv + c];
        w[r] = J[(r + 3) * nv + c];
      }
      cross3(wxm, w, sub_mcom + 3 * j);
      for (int r = 0; r < 3; r++)
        Jcom[r * nv + c] = (sub_mass[j] * v0[r] + wxm[r]) / total_mass;
    }
  }
  for (int i = 0; i < 3; i++) com[i] = sub_mcom[i] / total_mass;
  free(sub_mass);
  free(sub_mcom);
  return PyFloat_FromDouble(total_mass);
}

/* ------------------------------------------------------------------ */
/* SE(3) helpers exposed to the Python SE3 class                      */
/* ------------------------------------------------------------------ */

/* log6(R, p): twist of the transform (R, p), as (linear, angular) */
static PyObject *py_log6(PyObject *self, PyObject *args) {
  PyObject *o_R, *o_p;
  if (!PyArg_ParseTuple(args, "OO", &o_R, &o_p)) return NULL;
  double *R = as_f64(o_R, 9, "R");
  double *p = as_f64(o_p, 3, "p");
  if (!R || !p) return NULL;
  npy_intp dims[1] = {6};
  PyObject *out = new_f64(1, dims);
  if (!out) return NULL;
  log6((double *)PyArray_DATA((PyArrayObject *)out), R, p);
  return out;
}

/* Jlog6(R, p): Jacobian of log6 at the transform (R, p), as a 6x6 array */
static PyObject *py_jlog6(PyObject *self, PyObject *args) {
  PyObject *o_R, *o_p;
  if (!PyArg_ParseTuple(args, "OO", &o_R, &o_p)) return NULL;
  double *R = as_f64(o_R, 9, "R");
  double *p = as_f64(o_p, 3, "p");
  if (!R || !p) return NULL;
  npy_intp dims[2] = {6, 6};
  PyObject *out = new_f64(2, dims);
  if (!out) return NULL;
  jlog6((double *)PyArray_DATA((PyArrayObject *)out), R, p);
  return out;
}

/* exp6(v): transform (R, p) of the twist v = (linear, angular) */
static PyObject *py_exp6(PyObject *self, PyObject *args) {
  PyObject *o_v;
  if (!PyArg_ParseTuple(args, "O", &o_v)) return NULL;
  double *v = as_f64(o_v, 6, "v");
  if (!v) return NULL;
  double q[4], p[3], R[9];
  exp6_quat(q, p, v, v + 3);
  quat_to_mat(R, q);
  npy_intp rdims[2] = {3, 3};
  npy_intp pdims[1] = {3};
  PyObject *o_R = new_f64(2, rdims);
  PyObject *o_p = new_f64(1, pdims);
  if (!o_R || !o_p) {
    Py_XDECREF(o_R);
    Py_XDECREF(o_p);
    return NULL;
  }
  memcpy(PyArray_DATA((PyArrayObject *)o_R), R, sizeof(R));
  memcpy(PyArray_DATA((PyArrayObject *)o_p), p, sizeof(p));
  return Py_BuildValue("(NN)", o_R, o_p);
}

/* log3(R): rotation vector of the rotation matrix R */
static PyObject *py_log3(PyObject *self, PyObject *args) {
  PyObject *o_R;
  if (!PyArg_ParseTuple(args, "O", &o_R)) return NULL;
  double *R = as_f64(o_R, 9, "R");
  if (!R) return NULL;
  npy_intp dims[1] = {3};
  PyObject *out = new_f64(1, dims);
  if (!out) return NULL;
  double theta;
  log3((double *)PyArray_DATA((PyArrayObject *)out), &theta, R);
  return out;
}

/* exp3(w): rotation matrix of the rotation vector w */
static PyObject *py_exp3(PyObject *self, PyObject *args) {
  PyObject *o_w;
  if (!PyArg_ParseTuple(args, "O", &o_w)) return NULL;
  double *w = as_f64(o_w, 3, "w");
  if (!w) return NULL;
  double q[4], R[9];
  exp3_quat(q, w);
  quat_to_mat(R, q);
  npy_intp dims[2] = {3, 3};
  PyObject *out = new_f64(2, dims);
  if (!out) return NULL;
  memcpy(PyArray_DATA((PyArrayObject *)out), R, sizeof(R));
  return out;
}

/* ------------------------------------------------------------------ */
/* Module definition                                                  */
/* ------------------------------------------------------------------ */

static PyMethodDef methods[] = {
    {"forward_kinematics", py_forward_kinematics, METH_VARARGS,
     "Compute world placements of all joints, in place.\n"
     "\n"
     "Args:\n"
     "    jtype: Joint type of each joint (int32, one per joint).\n"
     "    parent: Parent joint index of each joint (int32).\n"
     "    idx_q: Index of each joint in the configuration vector (int32).\n"
     "    axis: Joint axis of each joint (float64, 3 per joint).\n"
     "    jp_rot: Rotation of each joint placement in its parent (9 each).\n"
     "    jp_trans: Translation of each joint placement in its parent "
     "(3 each).\n"
     "    q: Configuration vector.\n"
     "    oMi_rot: Output rotations of world joint placements (9 each).\n"
     "    oMi_trans: Output translations of world joint placements "
     "(3 each)."},
    {"joint_jacobians", py_joint_jacobians, METH_VARARGS,
     "Fill the full model Jacobian with world-frame columns, in place.\n"
     "\n"
     "Args:\n"
     "    jtype: Joint type of each joint (int32, one per joint).\n"
     "    idx_v: Index of each joint in the tangent vector (int32).\n"
     "    axis: Joint axis of each joint (float64, 3 per joint).\n"
     "    oMi_rot: Rotations of world joint placements (9 each).\n"
     "    oMi_trans: Translations of world joint placements (3 each).\n"
     "    J: Output model Jacobian, 6 x nv row-major.\n"
     "    nv: Dimension of the tangent space."},
    {"frame_placements", py_frame_placements, METH_VARARGS,
     "Compute world placements of all frames, in place.\n"
     "\n"
     "Args:\n"
     "    fparent: Parent joint index of each frame (int32).\n"
     "    fp_rot: Rotation of each frame placement in its joint (9 each).\n"
     "    fp_trans: Translation of each frame placement in its joint "
     "(3 each).\n"
     "    oMi_rot: Rotations of world joint placements (9 each).\n"
     "    oMi_trans: Translations of world joint placements (3 each).\n"
     "    oMf_rot: Output rotations of world frame placements (9 each).\n"
     "    oMf_trans: Output translations of world frame placements "
     "(3 each)."},
    {"frame_jacobian", py_frame_jacobian, METH_VARARGS,
     "Extract a frame Jacobian from the full model Jacobian, in place.\n"
     "\n"
     "Args:\n"
     "    J: Full model Jacobian, 6 x nv row-major, in world frame.\n"
     "    R_f: Rotation of the world placement of the frame (9 floats).\n"
     "    p_f: Translation of the world placement of the frame (3 floats).\n"
     "    support: Tangent-space columns of the joints supporting the frame\n"
     "        (int32); other columns of the output are left at zero.\n"
     "    rf: Reference frame: 0 world, 1 local, 2 local-world-aligned.\n"
     "    out: Output frame Jacobian, 6 x nv row-major.\n"
     "    nv: Dimension of the tangent space."},
    {"integrate", py_integrate, METH_VARARGS,
     "Integrate a tangent-space velocity into a configuration, in place.\n"
     "\n"
     "Args:\n"
     "    jtype: Joint type of each joint (int32, one per joint).\n"
     "    idx_q: Index of each joint in the configuration vector (int32).\n"
     "    idx_v: Index of each joint in the tangent vector (int32).\n"
     "    q: Configuration vector to integrate from.\n"
     "    v: Tangent-space velocity to integrate.\n"
     "    qout: Output configuration vector, same size as q."},
    {"difference", py_difference, METH_VARARGS,
     "Tangent-space difference between two configurations, in place.\n"
     "\n"
     "Args:\n"
     "    jtype: Joint type of each joint (int32, one per joint).\n"
     "    idx_q: Index of each joint in the configuration vector (int32).\n"
     "    idx_v: Index of each joint in the tangent vector (int32).\n"
     "    q0: Configuration to start from.\n"
     "    q1: Configuration to go to.\n"
     "    dout: Output velocity such that integrate(q0, dout) is q1."},
    {"d_difference", py_d_difference, METH_VARARGS,
     "Jacobian of the configuration difference, in place.\n"
     "\n"
     "Args:\n"
     "    jtype: Joint type of each joint (int32, one per joint).\n"
     "    idx_q: Index of each joint in the configuration vector (int32).\n"
     "    idx_v: Index of each joint in the tangent vector (int32).\n"
     "    q0: Configuration to start from.\n"
     "    q1: Configuration to go to.\n"
     "    arg: Differentiate with respect to q0 (0) or q1 (1).\n"
     "    Jout: Output Jacobian, nv x nv row-major, block-diagonal.\n"
     "    nv: Dimension of the tangent space."},
    {"center_of_mass", py_center_of_mass, METH_VARARGS,
     "Compute the whole-robot center of mass in the world frame.\n"
     "\n"
     "The inertia attached to the universe joint is not included.\n"
     "\n"
     "Args:\n"
     "    parent: Parent joint index of each joint (int32).\n"
     "    mass: Mass of the body attached to each joint.\n"
     "    lever: Center of mass of each body in its joint frame (3 each).\n"
     "    oMi_rot: Rotations of world joint placements (9 each).\n"
     "    oMi_trans: Translations of world joint placements (3 each).\n"
     "    com_out: Output center of mass in the world frame (3 floats).\n"
     "\n"
     "Returns:\n"
     "    Total mass of the robot."},
    {"com_jacobian", py_com_jacobian, METH_VARARGS,
     "Compute the center-of-mass Jacobian in the world frame.\n"
     "\n"
     "Args:\n"
     "    jtype: Joint type of each joint (int32, one per joint).\n"
     "    parent: Parent joint index of each joint (int32).\n"
     "    idx_v: Index of each joint in the tangent vector (int32).\n"
     "    mass: Mass of the body attached to each joint.\n"
     "    lever: Center of mass of each body in its joint frame (3 each).\n"
     "    oMi_rot: Rotations of world joint placements (9 each).\n"
     "    oMi_trans: Translations of world joint placements (3 each).\n"
     "    J: Full model Jacobian, 6 x nv row-major, in world frame.\n"
     "    Jcom: Output center-of-mass Jacobian, 3 x nv row-major.\n"
     "    com_out: Output center of mass in the world frame (3 floats).\n"
     "    nv: Dimension of the tangent space.\n"
     "\n"
     "Returns:\n"
     "    Total mass of the robot."},
    {"log6", py_log6, METH_VARARGS,
     "Logarithm map of a rigid transform.\n"
     "\n"
     "Args:\n"
     "    R: Rotation of the transform (9 floats, row-major).\n"
     "    p: Translation of the transform (3 floats).\n"
     "\n"
     "Returns:\n"
     "    Twist of the transform, as (linear[3], angular[3])."},
    {"Jlog6", py_jlog6, METH_VARARGS,
     "Jacobian of the SE3 logarithm at a rigid transform.\n"
     "\n"
     "Args:\n"
     "    R: Rotation of the transform (9 floats, row-major).\n"
     "    p: Translation of the transform (3 floats).\n"
     "\n"
     "Returns:\n"
     "    Jacobian of log6 at the transform, as a 6 x 6 array."},
    {"exp6", py_exp6, METH_VARARGS,
     "Exponential map of a twist.\n"
     "\n"
     "Args:\n"
     "    v: Twist, as (linear[3], angular[3]).\n"
     "\n"
     "Returns:\n"
     "    Pair (R, p) of the rotation matrix and translation of the\n"
     "    corresponding rigid transform."},
    {"log3", py_log3, METH_VARARGS,
     "Logarithm map of a rotation matrix.\n"
     "\n"
     "Args:\n"
     "    R: Rotation matrix (9 floats, row-major).\n"
     "\n"
     "Returns:\n"
     "    Rotation vector, whose norm is the rotation angle."},
    {"exp3", py_exp3, METH_VARARGS,
     "Exponential map of a rotation vector.\n"
     "\n"
     "Args:\n"
     "    w: Rotation vector, whose norm is the rotation angle.\n"
     "\n"
     "Returns:\n"
     "    Corresponding 3 x 3 rotation matrix."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_kinematics_c",
    "Numerical kernels for pinker.kinematics (C implementation)", -1, methods};

/* Module initialization: bring in the NumPy C API, then create the module */
PyMODINIT_FUNC PyInit__kinematics_c(void) {
  import_array();
  return PyModule_Create(&module);
}
