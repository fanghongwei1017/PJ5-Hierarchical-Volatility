import math
import numpy as np
from scipy import special
from scipy.optimize import minimize_scalar

def grid(n, beta, size=65536, width_factor=1.0, engine='convolution'):
    """Centered, probability-normalized FFT avoids powers of 1/dy overflowing.

    For arbitrary beta the domain follows the exact log moments, preventing
    circular wraparound of very deep low-beta cascades. Characteristic inversion
    is an independent check using the analytic log-Gamma characteristic function.
    """
    mu1 = math.log(beta) - special.digamma(beta + 1)
    var1 = special.polygamma(1, beta + 1)
    mu = n * mu1
    half = max(45., 12 * math.sqrt(n * var1)) * width_factor
    h = 2 * half / size
    u = -half + h * np.arange(size)
    y = mu + u
    if engine == 'convolution':
        z = mu1 + u
        lg = (beta + 1) * math.log(beta) - special.gammaln(beta + 1)
        lg = lg - (beta + 1) * z - beta * np.exp(np.clip(-z, -745, 700))
        mass1 = np.exp(lg) * h
        mass1 /= mass1.sum()
        spectrum = np.fft.rfft(np.fft.ifftshift(mass1))
        density = np.fft.fftshift(np.fft.irfft(spectrum ** n, n=size)) / h
    else:
        w = 2 * np.pi * np.fft.rfftfreq(size, d=h)
        if n>=32 and beta>=20:
            # Avoid subtracting two large log-Gamma values: use the convergent
            # Taylor series of the centered analytic characteristic function.
            # All retained frequencies obey |w|/shape <= 1/2. The omitted
            # spectral edge must already be below exp(-40).
            cap=min(.5*(beta+1),math.sqrt(200/(n*var1)))
            keep=w<=cap
            z=1j*w[keep]
            logphi=np.zeros(len(z),dtype=complex)
            for k in range(2,49):
                logphi += (n*special.polygamma(k-1,beta+1)/math.factorial(k))*z**k
            assert logphi[-1].real < -40
            spectrum=np.zeros(len(w),dtype=complex)
            spectrum[keep]=np.exp(logphi)
        else:
            logphi = n * (-1j*w*math.log(beta) + special.loggamma(beta+1+1j*w)
                          - special.gammaln(beta+1)) + 1j*w*mu
            # At lower depths keep the exact log-Gamma tail, but evaluate the
            # low-frequency centered part by its convergent Taylor series too.
            keep=w<=.5*(beta+1)
            z=1j*w[keep]
            centered=np.zeros(len(z),dtype=complex)
            for k in range(2,49):
                centered += (n*special.polygamma(k-1,beta+1)/math.factorial(k))*z**k
            logphi[keep]=centered
            spectrum=np.exp(logphi)
        density = np.fft.fftshift(np.fft.irfft(spectrum, n=size)) / h
    # Fourier roundoff in far negative log-variance tails can be magnified at x=0.
    density = np.maximum(density, 0)
    cut = density.max() * 1e-13
    removed = float(density[density < cut].sum() * h)
    density[density < cut] = 0
    density /= density.sum() * h
    return y, density, {'N':n,'beta':beta,'removed_roundoff_mass': removed, 'half_width': half,
                        'exact_log_mean': float(mu), 'exact_log_variance': float(n*var1)}


def limit_grid(tau, size=65536, width_factor=1.):
    half = max(45., 12*math.sqrt(tau)) * width_factor
    y = -tau/2 - half + 2*half/size*np.arange(size)
    g = np.exp(-.5*(y+tau/2)**2/tau) / math.sqrt(2*math.pi*tau)
    return y, g, {'tau':tau,'exact_log_mean': -tau/2, 'exact_log_variance': tau}


def logpdf(x, g, target=4096):
    y, dens = g[:2]
    stride = max(1, len(y)//target)
    ys, ds = y[::stride], dens[::stride]
    good = ds > 0
    logw = np.log(ds[good]) + math.log(y[1]-y[0]) + math.log(stride)
    yy = ys[good]
    invv = np.exp(np.clip(-yy, -745, 700))
    base = logw - .5*(math.log(2*math.pi)+yy)
    x = np.asarray(x)
    out = np.empty(len(x))
    for start in range(0, len(x), 256):
        a = x[start:start+256]
        terms = -.5*a[:,None]**2*invv + base
        out[start:start+len(a)] = special.logsumexp(terms, axis=1)
    # Exact negative-half moment: FFT noise in the far-left background tail
    # is amplified at zero. Also covers an odd histogram's rounded zero center.
    zeros=np.abs(x)<1e-14
    meta=g[2]
    if np.any(zeros):
        if 'N' in meta:
            b=meta['beta']
            out[zeros]=meta['N']*(math.log(special.poch(b+1,.5))-.5*math.log(b))-.5*math.log(2*math.pi)
        else:
            out[zeros]=3*meta['tau']/8-.5*math.log(2*math.pi)
    assert np.isfinite(out).all()
    return out


class Objective:
    def __init__(self, x, v, spec):
        self.x, self.v, self.spec = x, v, spec
        self.cx, self.px = spec.histogram(spec.x_edges, x)
        self.cv = np.histogram(np.log(v[v>0]), bins=spec.v_edges_log)[0]
        self.pv = self.cv / (self.cv.sum()*spec.v_widths_log*spec.v_centers)
        self.denom = 2*len(x)-len(v)

    def evaluate(self, g, target=4096):
        lp = logpdf(self.spec.x_centers, g, target)
        f = np.interp(np.log(self.spec.v_centers), g[0], g[1], left=0,right=0)/self.spec.v_centers
        lf = np.log(np.maximum(f,1e-300))
        ix, iv = self.cx>0, self.cv>0
        # Normalize counts BEFORE multiplying very large losses. This preserves
        # Eq.24 but avoids overflow in deliberately poor low-beta/deep candidates.
        tx = float(np.sum((self.cx[ix]/self.denom)*np.abs(lp[ix]-np.log(self.px[ix]))))
        tv = float(np.sum((self.cv[iv]/self.denom)*np.abs(lf[iv]-np.log(self.pv[iv]))))
        return {'error':tx+tv, 'term_returns':tx, 'term_background':tv}


def fit(obj, n=None, size=65536, target=4096, width_factor=1., coarse_points=49):
    lo, hi = ((.0001,100.) if n is None else (.05,max(20.,4.*n)))
    factory = (lambda b: limit_grid(b,size,width_factor)) if n is None else (lambda b: grid(n,b,size,width_factor))
    calls = []
    def fun(z):
        b = math.exp(z)
        e = obj.evaluate(factory(b),target)['error']
        calls.append([b,e])
        return e
    zz = np.linspace(math.log(lo),math.log(hi),coarse_points)
    ee = np.array([fun(z) for z in zz])
    candidates = [(ee[0],zz[0]),(ee[-1],zz[-1])]
    local = []
    for i in range(1,len(zz)-1):
        if ee[i] <= ee[i-1] and ee[i] <= ee[i+1] and (ee[i] < ee[i-1] or ee[i] < ee[i+1]):
            r = minimize_scalar(fun,bounds=(zz[i-1],zz[i+1]),method='bounded',options={'xatol':1e-7,'maxiter':200})
            local.append({'success':bool(r.success),'iterations':int(r.nit)})
            candidates.append((float(r.fun),float(r.x)))
    val,z = min(candidates)
    b = math.exp(z)
    return {'N':n,'tau' if n is None else 'beta':b,**obj.evaluate(factory(b),target),
            'parameter_bounds':[lo,hi], 'parameter_boundary':bool(b <= lo*1.02 or b>=hi*.98),
            'optimizer_success':bool(local) and all(r['success'] for r in local),
            'local_searches':local,'evaluations':calls,'grid_diagnostics':factory(b)[2]}


