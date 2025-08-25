import time  
import numpy as np
import streamlit as st
from NN_emulator import NNEmulator
import plotly.graph_objects as go
import camb

st.set_page_config(page_title='P(k,z) Emu. Demo.', layout='wide')

# ---------- These will cache items for faster re-loading of page ----------
@st.cache_data
def load_data():
    data = np.load('train_val_data.npz')
    pk   = np.load('pk_example.npz')
    val_errs = np.load('val_errs.npy')

    return {
        "ks": data["ks"],                # (k_len)
        "zs": data["zs"],                # (z_len)
        "pcC": data["pca_components"],   # (N_pc, k_len*z_len)
        "pcMean": data["pca_mean"],      # (k_len*z_len)
        "pk_l": pk["lin"],               # example P(k) for Fig. 1
        "pk_nl": pk["nonlin"],
        "val_errs": val_errs             # validation errors for Fig. 2
    }

@st.cache_resource
def load_emulator(model_path: str = 'NN_baseline.keras', data=None):
    emu = NNEmulator(
        N_pc=len(data['pcC']),
        hidden=512,
        depth=3,
        lr=1e-3,
        decayevery=500,
        decayrate=0.5,
        pca_components=data['pcC'],
        pca_mean=data['pcMean']
    )
    emu.load(model_path)
    return emu

@st.cache_data
def get_linear_pk(h, Omega_b, Omega_c, As, ns, w,tau = 0.078):
    cosmology = camb.set_params(# Background
                    H0 = 100*h, ombh2=Omega_b*h**2, omch2=Omega_c*h**2,
                    TCMB = 2.7255,
                    # Dark Energy
                    dark_energy_model='fluid', w = w,
                    # Neutrinos
                    nnu=3.046, mnu = 0.058, num_nu_massless = 0, num_massive_neutrinos = 3,
                    # Initial Power Spectrum
                    As = As, ns = ns, tau = 0.0543,
                    YHe = 0.246, WantTransfer=True)
    pk = camb.get_matter_power_interpolator(
            cosmology,
            nonlinear=False,
            hubble_units=True, 
            k_hunit=True,   
            kmax=3.14159,
            zmax=2.0)
    pk = pk.P(zs,ks)
    return pk

@st.cache_data
def get_Q_cached(theta, z_len, k_len):
    return emu.predict_Q(theta, out_shape=(z_len, k_len))

# ---------- Loading data ----------------
data = load_data()
emu = load_emulator(data=data)

ks = data['ks']
zs = data['zs']
val_errs = data['val_errs']
pk_l = data['pk_l']
pk_nl = data['pk_nl']

k_len = len(ks)
z_len = len(zs)
N_val = val_errs.shape[0]


st.title('Matter Power Spectrum Emulator Demo')

# ---------- Science background ------------------
with st.expander(r'**Scientific Context and Significance**', expanded=False):
    st.write(r'''The matter power spectrum $P(k,z)$ is a quantity that encodes the statistical 
        distribution and structure of matter across different length scales, at different times, throughout the Universe. 
        Technically, the power spectrum is a function of the Fourier wavenumber $k$, and the redshift $z$. Effectively, 
        $k$ sets the spatial scales in the Universe (from largest to smallest), and $z$ sets the particular time in the 
        Universe\'s history (also measured from largest to smallest, so that $z=0$ corresponds to today). The matter power 
        spectrum also depends on the parameters of one\'s cosmological model. For example, $H_0$ which encodes the Universe's 
        expansion rate, or $\Omega_m$ which specifies what fraction of the Universe's total energy density is in the form of 
        matter. The matter power spectrum is a fundamental quantity in Cosmology, and is an essential link for comparing 
        theoretical predictions to experimental observations. Unfortunately, it can also prove difficult to model how such 
        simple parameters translate into the rich structure of galaxies and clusters in the Universe. While it is relatively easy 
        to compute the first order (\"linear\") approximation to the power spectrum $P_L(k,z)$ using numerical equation solvers, 
        as shown below the linear approximation fails on the smaller scales$^1$ of the Universe (high $k$) where the interactions 
        of matter are more complex.''') 
    st.write(r'''This problem is particularly urgent at the present moment, as the next generation of experiments called \"Stage-IV\" 
        such as LSST and Euclid are beginning to collect data that will measure these problematic small-scales with high accuracy.
        If we want this data to give insights into the Universe's true cosmological parameters, we need a fast$^2$ way to model 
        $P(k,z)$ that is accurate on small scales. $N$-body simulations, which simulate the interactions of billions of matter 
        particles clustering under their mutual gravitational attraction, are the best way to model $P(k,z)$ on small scales, but 
        they typically take thousands of CPU hours and terabytes of memory. This is the reason the Neural Network presented below, 
        which can learn from a training set of simulations and reproduce their results in well under a second on a simple laptop.''')


    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ks, y=pk_l[0],
        mode='lines',
        name='Linear Approximation',
        line=dict(color='orange', dash='dot')
    ))
    fig1.add_trace(go.Scatter(
        x=ks, y=pk_nl[0],
        mode='lines',
        name='Exact',
        line=dict(color='blue')
    ))
    fig1.update_layout(
        xaxis=dict(
            title='k [h/Mpc]',
            type='log',
            tickmode='array',
            tickvals=[1e-2, 1e-1, 1],
            ticktext=['10<sup>-2</sup>', '10<sup>-1</sup>', '10<sup>0</sup>'] ,
            showline=True,        
            linecolor='black',    
            linewidth=1,          
            mirror=True           
        ),
        yaxis=dict(
            title='P(k,z=0) [Mpc/h]^3',
            type='log',
            tickmode='array',
            tickvals=[1, 1e1, 1e2, 1e3, 1e4],
            ticktext=['10<sup>0</sup>', '10<sup>1</sup>', '10<sup>2</sup>', '10<sup>3</sup>', '10<sup>4</sup>'],
            showline=True,        
            linecolor='black',    
            linewidth=1,          
            mirror=True    
        ),
        legend=dict(
            title='Spectrum Type',
            x=0.98,   
            y=0.98,     
            xanchor='right',
            yanchor='top',
            font=dict(size=14),      
            title_font=dict(size=16),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=10, r=10, t=30, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.write(r'''We see from the plot that after scales of order $k \sim 10^{-1}$ $h$Mpc$^{-1}$ the linear approximation is no longer accurate.
        Nonlinear behavior grows with time as matter in the Universe clusters together, so $z=0$ is plotted to show maximum discrepancy.''')

    st.caption(r'$^1$ \"Small-scales\" in Cosmology are pretty large, about the scale of galaxy clusters (1 Mpc $\approx$ 3 million light years).')
    st.caption(r'''$^2$ Typical Cosmological analyses use Monte Carlo methods and require $10^5$-$10^6$ predictions of the power spectrum to find 
        the best-fit parameters, so we need to model $P(k,z)$ in at most a few seconds.''')


# ------------ NN Architecture --------------
with st.expander(r'**Neural Network Architecture**', expanded=False):
    st.write(r'''For this demo I use the $w$CDM cosmological model which can be specified by 6 cosmological parameters: ($\Omega_m$, 
        $\Omega_b$, $n_s$, $A_s$, $h$, $w$). The model must map this 6D input to a surface of 2 variables P(k,z). This is done by predicting the 
        function values over a fixed 2D grid of (k,z) values of shape (512, 35), which is flattened to a 1D vector of shape (17920). 
        However, due to strong correlations between the bins of the function, PCA is effective at reducing the dimensionality of the 
        output so that the model must only predict 50 PC amplitudes. The model is trained on the following nonlinear correction to the power 
        spectrum $Q(k,z) \equiv \log(P(k,z)/P_L(k,z))$ as it helps normalizes the data and eliminates linear features which there is no 
        need to learn. At prediction time, the full dimension is restored from the PC amplitudes, the linear approximation $P_L(k,z)$ is 
        computed numerically and the nonlinear correction is applied to recover $P(k,z)$. Due to the high computational expense of the 
        training simulations, we are limited to 500 training samples.''')
    st.write(r'''I use a fully connected MLP ($6 \rightarrow 512 \rightarrow 512 \rightarrow 512 \rightarrow 50$) with input dimension 6 and output 
        dimension 50, with a custom activation specified by:''')
    st.latex(r'y^{m+1}_n = \left[\gamma^m_n + (1 - \gamma^m_n)\frac{1}{1 + e^{-\beta^m_n y^m_n}}\right]\tilde{y}^m_n ,')
    st.write(r'''where $\gamma^m_n$ and $\beta^m_n$ are learnable parameters, $y^{m}_n$ is the $n$th neuron from the $m$th layer, and 
        the tilde in $\tilde{y}^{m}_n$ denotes that weights and biases have been apllied. This activation effectively serves as a combination of 
        a linear pass-through and a sigmoid gate, allowing for adaptive nonlinearity. The architecture is summarized below:''')

    dot = r'''digraph G {
          rankdir=LR; nodesep=0.5; ranksep=0.4;
          node [fontname='Helvetica', color='#444', fixedsize=true];

          input  [shape=box, style=filled, width=0.55, height=0.5, fillcolor='#f6f8fa', label='Input\n(6)'];
          h1     [shape=box, style=filled, width=0.55, height=1.6, fillcolor='#cce5ff', label='Hidden 1\n(512)'];
          h2     [shape=box, style=filled, width=0.55, height=1.6, fillcolor='#cce5ff', label='Hidden 2\n(512)'];
          h3     [shape=box, style=filled, width=0.55, height=1.6, fillcolor='#cce5ff', label='Hidden 3\n(512)'];
          output [shape=box, style=filled, width=0.7, height=1.0, fillcolor='#d5f5e3', label='Output\nPC\namplitudes\n(50)'];

          act1 [shape=ellipse, width=0.25, height=0.25, label='act.'];
          act2 [shape=ellipse, width=0.25, height=0.25, label='act.'];
          act3 [shape=ellipse, width=0.25, height=0.25, label='act.'];

          input -> h1 -> act1 -> h2 -> act2 -> h3 -> act3 -> output;
        }'''

    st.graphviz_chart(dot, use_container_width=True)


# ---------- Validation ---------------
with st.expander(r'**Validation**', expanded=False):

    st.write(r'''Below the NN predictions were compared to a validation set of 100 simulations, chosen by LH sampling. These validation 
        samples were used for early stopping during training. As before, $z=0$ is plotted for maximum error as difficult-to-learn nonlinear 
        structure in the Universe grows over time.''')

    fig2 = go.Figure()
    for i in range(N_val):
        fig2.add_trace(go.Scatter(
            x=ks[1:],
            y=100*val_errs[i,1:],
            mode='lines',
            line=dict(width=1),
            showlegend=False 
            )
        )

    fig2.update_xaxes(title='k [h/Mpc]',
            type='log',
            tickmode='array',
            tickvals=[1e-2, 1e-1, 1],
            ticktext=['10<sup>-2</sup>', '10<sup>-1</sup>', '10<sup>0</sup>'] ,
            showline=True,        
            linecolor='black',    
            linewidth=1,          
            mirror=True   
            )
    fig2.update_yaxes(title='% error in P(k,z=0)')

    st.plotly_chart(fig2, use_container_width=True)

    st.write(r'''We see that 93% of samples are within 0.3% error of validation simulations, and 100% are wthin 0.61% error.''')

# ---------- DEMO --------------------
st.write(r'**Emulator Demo**')
st.write(r'Adjust the 6 cosmological parameters of the $w$CDM model below for an interactive plot of the $P(k,z)$ surface.')

# -------- Input sliders -------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(r'**$\Omega_m$**')
    Om = st.slider('', 0.24, 0.40, 0.32, 0.001, key='Om')

    st.markdown(r'**$\Omega_b$**')
    Ob = st.slider('', 0.04, 0.06, 0.05, 0.001, key='Ob')

with col2:
    st.markdown(r'**$n_s$**')
    ns = st.slider('', 0.92, 1.00, 0.96, 0.001, key='ns')

    st.markdown(r'**$A_s \times 10^9$**')
    As_scaled = st.slider('', 1.7, 2.5, 2.1, 0.01, key='As_scaled')

with col3:
    st.markdown(r'**$h$**')
    h = st.slider('', 0.61, 0.73, 0.67, 0.001, key='h')

    st.markdown(r'**$w$**')
    w = st.slider('', -1.30, -0.70, -1.00, 0.001, key='w')

# ---------- Normalize inputs ----------
# Order: [Om, Ob, ns, As_scaled, h, w]
param_mins = np.array([0.24, 0.04, 0.92, 1.70, 0.61, -1.30], dtype=np.float32)
param_maxs = np.array([0.40, 0.06, 1.00, 2.50, 0.73, -0.70], dtype=np.float32)

theta = np.array([Om, Ob, ns, As_scaled, h, w], dtype=np.float32)
theta = (theta - param_mins) / (param_maxs - param_mins)
theta = theta.reshape(1, -1)

# ---------- Inference ----------
start = time.perf_counter()
Q = get_Q_cached(theta, z_len, k_len)
elapsed_ms = (time.perf_counter() - start) * 1e3
st.info(f'NN inference time: **{elapsed_ms:.1f} ms** on CPU')

start = time.perf_counter()
pk_l = get_linear_pk(h, Om, Om-Ob, As_scaled*1e-9, ns, w,tau = 0.078)
elapsed_ms = (time.perf_counter() - start) * 1e3
st.info(f'Linear calculation time: **{elapsed_ms:.1f} ms** on CPU')


# ---------- Interactive plot ----------
status = st.empty()
status.write('Now plotting...')

K, Z = np.meshgrid(ks, zs)
fig3 = go.Figure(data=[go.Surface(x=K, y=Z, z=pk_l*np.exp(Q), colorscale='Viridis')])

fig3.update_layout(
    scene=dict(
        xaxis=dict(
            title='k [h/Mpc]',
            type='log',
            tickmode='array',
            tickvals=[1e-2, 1e-1, 1],
            ticktext=['10<sup>-2</sup>', '10<sup>-1</sup>', '10<sup>0</sup>']
        ),
        yaxis=dict(
            title='z'),
        zaxis=dict(
            title='P(k,z)  [Mpc/h]^3',
            range=[0, 29000])
    ),
    scene_camera=dict(eye=dict(x=0.1, y=-2.3, z=0.5)),
    margin=dict(l=0, r=0, b=0, t=30)
)

st.plotly_chart(fig3, use_container_width=True)
status.empty()

status.write('''Congrats, you just modeled the nonlinear matter power spectrum with simulation-level accuracy, 
    without needing a supercomputer or thousands of CPU hours!''')


# ------------- Acknowledgements ------------
with st.expander('Acknowledgements'):
    st.write('''This demo is based largely on work from my PhD, published in 
        [**this paper**](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.083529), with a publicly available pdf 
        [**here**](https://arxiv.org/abs/2404.12344). I would like to thank my co-authors for their invaluable contributions throughout 
        this project, and all parties which made that initial work possible. More to come!

        (Demo created by Jonathan Gordon, adapted from our published work.)''')
