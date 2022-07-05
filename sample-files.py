import h5py
import pandas as pd
import glob
import numpy as np
from scipy.stats import gaussian_kde



cols = ['luminosity_distance', 'mass_1'] # Only taking directly observable values
n_runs = 100   # number of random plots to make
n_m = 100    # resolution of mass points to find max over
n_mc = 15     # number of points to fit for the MCMC
m_cut = 25   # M_sun

mass_array = np.zeros((n_runs, n_mc))

d_ls = np.linspace(0, 5000, n_mc)

for run in range(n_runs):
    df = False #janky method, alternative would be strongly appreciated

    for f in glob.glob('O3b-data/*mixed_cosmo.h5'): #all the mixed cosmology files
        with h5py.File(f, 'r') as hf:
            post = np.array(hf['C01:Mixed/posterior_samples']) #posterior to sample from
            
            # if definitely not a mass bump BH or too far away...
            if (np.median(post['mass_1']) < m_cut) \
                or (np.median(post['luminosity_distance']) > 10000):
                continue
            data_array = np.random.choice(post, 1) #take one random sample from posterior
            if type(df) == bool: #janky method continued
                df = pd.DataFrame(data_array)[cols] #only want M_1 and d_L
            else:
                if np.median(data_array['mass_1']) > m_cut:
                    df = pd.concat([df, pd.DataFrame(data_array)], ignore_index=True)[cols]

    for f in glob.glob('../LIGO-O3a-Posterior/all_posterior_samples/*comoving.h5'): #same as above but O3a
        with h5py.File(f, 'r') as hf:
            post = np.array(hf['PublicationSamples/posterior_samples'])
            if np.median(post['mass_1']) < m_cut:
                continue
            data_array = np.random.choice(post, 1)
            if type(df) == bool: #janky method continued
                df = pd.DataFrame(data_array)[cols]
            else:
                if np.median(data_array['mass_1']) > m_cut:
                    df = pd.concat([df, pd.DataFrame(data_array)], ignore_index=True)[cols]

    arr = np.array([df['luminosity_distance'], df['mass_1']])
    kde = gaussian_kde(arr) #KDE of d_L vs m_1


    dlmin = 0
    dlmax=max(d_ls)
    mmin = 0
    mmax = max(df['mass_1'])

    DL, M = np.mgrid[dlmin:dlmax:n_mc*1j, mmin:mmax:n_m*1j]
    positions = np.vstack([DL.ravel(), M.ravel()])
    K = np.reshape(kde(positions).T, DL.shape)


    ms = np.linspace(mmin, mmax, n_m)

    for i, dl in enumerate(K):
        mass_array[run, i] = ms[np.argmax(dl)]

df = pd.DataFrame(mass_array)
file_name = f"mass_array_{n_mc}.csv"
df.to_csv(file_name)