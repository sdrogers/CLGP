import numpy as np
import pylab as plt
import copy
import warnings


def create_dataset(N, n_data):
    true_vals = np.random.rand(N)
    true_vals.sort()
    data_1_idx = np.random.permutation(range(N))[:n_data]
    data_2_idx = np.random.permutation(range(N))[:n_data]
    data_1_idx.sort()
    data_2_idx.sort()
    opt = len(set(data_1_idx).intersection(set(data_2_idx)))
    print("Size of Overlap between datasets = {}".format(opt))
    return true_vals, data_1_idx, data_2_idx


def add_frag_events(n_data, frag_prob_1):
    frag_1 = []
    for i in range(n_data):
        if np.random.rand() <= frag_prob_1:
            frag_1.append(1)
        else:
            frag_1.append(0)
    return frag_1


def plot_frag_dataset(true_vals, data_idx, frag_1, col='g'):
    plt.xlabel('Time of Observation')
    plt.ylabel('Observation Index')
    for i, idx in enumerate(data_idx):
        if frag_1[i] == 1:
            m = 'x'
        else:
            m = 'o'
        plt.scatter(true_vals[idx], idx, marker=m, c=col)
    plt.plot([], [], 'gx', label='Object with Additional Information')
    plt.plot([], [], 'go', label='Object with no Additional Information')
    plt.legend()
    plt.show()


def create_drift(true_vals, alpha, gam):
    N = len(true_vals)
    K = np.zeros((N, N), np.double)
    for n in range(N):
        for m in range(N):
            K[n,m] = alpha*np.exp(-(1./gam)*(true_vals[n]-true_vals[m])**2)
    true_offset_function = np.random.multivariate_normal(np.zeros(N),K)
    return K, true_offset_function


def create_observed_datasets(true_vals, data_1_idx, data_2_idx, true_offset_function, noise_ss, alpha, gam):
    n_data = len(true_vals[data_1_idx])
    observed_1 = true_vals[data_1_idx]
    observed_2 = true_vals[data_2_idx] + true_offset_function[data_2_idx] + np.random.normal(n_data)*np.sqrt(noise_ss)
    main_K = np.zeros((n_data, n_data), np.double)
    for n in range(n_data):
        for m in range(n_data):
            main_K[n, m] = alpha*np.exp((-1./gam)*(observed_2[n] - observed_2[m])**2)
    return observed_1, observed_2, main_K


def plot_datasets(dataset1, dataset2, dataset1_idx, dataset2_idx, true_matching=False, matches=None,
                  confirmed_matches=None, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.plot(dataset1, dataset1_idx, 'go', label='dataset1')
    plt.plot(dataset2, dataset2_idx, 'ko', label='dataset2')
    plt.xlabel('Time of Observation')
    plt.ylabel('Observation Index')
    if true_matching:
        unique_ids = np.unique(np.append(dataset1_idx, dataset2_idx))
        plt.plot([], [], 'k', label="True Matches")
        for idx in unique_ids:
            if idx in dataset1_idx and idx in dataset2_idx:
                plt.plot([dataset1[dataset1_idx == idx], dataset2[dataset2_idx == idx]], [idx,idx], 'k')
    if matches is not None:
        plt.plot([], [], 'b', label="Confirmed Matches")
        plt.plot([], [], 'b--', label="Unconfirmed Matches")
        plt.plot([], [], 'r', label="Incorrect Matches")
        for m in matches:
            if confirmed_matches is not None and m in confirmed_matches:
                # this is a confirmed match
                plt.plot([dataset1[m[0]], dataset2[m[1]]],[dataset1_idx[m[0]], dataset2_idx[m[1]]], 'b')
            elif dataset1_idx[m[0]] == dataset2_idx[m[1]]:
                # this is a correct, but unconfirmed match
                plt.plot([dataset1[m[0]], dataset2[m[1]]],[dataset1_idx[m[0]], dataset2_idx[m[1]]], 'b--')
            else:
                # incorrect match
                plt.plot([dataset1[m[0]], dataset2[m[1]]],[dataset1_idx[m[0]], dataset2_idx[m[1]]], 'r')
    plt.legend(loc='lower right', prop={"size":12})


def closest_match(peaks1, peaks2, confirmed_matches, queried_points, frag_1, max_rt=0.1, predictions=None):
    matches = copy.deepcopy(confirmed_matches)
    if len(matches) > 0:
        used1, used2 = zip(*matches)
        used1 = set(used1)
        used2 = set(used2)
    else:
        used1 = set()
        used2 = set()
    for i, r in enumerate(peaks2):

        r2 = r
        if predictions is not None:
            r2 += predictions[i]

        if i in used2:  # if this one is in one of the confirmed matches, we don't need to deal with it
            continue
        else:
            # compute all distances
            pi = list(zip(peaks1, range(len(peaks1)), [abs(p - r2) for p in peaks1]))
            # keep only those below the max time shift for matching
            pi = list(filter(lambda x: x[2] <= max_rt, pi))
            # find the closest that is not already used .... and
            # if the set 2 point (i) has been queried before then if we get to here it wasn't matched
            # so it cannot be matched against something for which we have the additional info
            closest_dist = 1e6
            best_pos = -1
            for idx, _ in enumerate(pi):
                if pi[idx][1] not in used1 and pi[idx][2] <= closest_dist and not (
                        i in queried_points and frag_1[idx] == 1):
                    best_pos = idx
                    closest_dist = pi[idx][2]
            if best_pos > -1:
                matches.append((pi[best_pos][1], i))
                used1.add(pi[best_pos][1])
                used2.add(i)
                assert pi[best_pos][2] <= max_rt
    return matches


def plot_match(peaks1,peaks2,frag_1,matches,confirmed_matches,relative=True,predictions = None,truth=None):
    # plt.figure()
    if not relative:
        plt.plot(peaks2,np.zeros_like(peaks2),'ro')
        plt.plot(np.zeros_like(peaks1),peaks1,'ro')
        p1 = []
        for i,f in enumerate(frag_1):
            if f == 1:
                p1.append(peaks1[i])
        plt.plot(np.zeros_like(p1),p1,'ko')

    for a,(i,j) in enumerate(matches):
        col = 'ro' # default
        if truth is not None:
            if truth[a] == 1:
                col = 'bo' # match is correct, so make it blue
        if (i,j) in confirmed_matches: # if this is a confirmed one
            if not relative:
                plt.plot(peaks2[j],peaks1[i],col,markersize=15)
            else: # plotting difference not absolute time
                plt.plot(peaks2[j],peaks1[i]-peaks2[j],col,markersize=15)
                if predictions is not None:
                    plt.plot(peaks2[j],predictions[j],'ko',markersize=3) # plot a point where the current model prediction is
                    plt.plot([peaks2[j],peaks2[j]],[predictions[j],peaks1[i]-peaks2[j]],'k',color=[0.8,0.8,0.8])
        else:
            if not relative:
                plt.plot(peaks2[j],peaks1[i],col)
            else:
                plt.plot(peaks2[j],peaks1[i]-peaks2[j],col)
                if not predictions is None:
                    plt.plot(peaks2[j],predictions[j],'ko',markersize=3)
                    plt.plot([peaks2[j],peaks2[j]],[predictions[j],peaks1[i]-peaks2[j]],'k',color=[0.8,0.8,0.8])
        if not relative:
            plt.plot([peaks2[j],peaks2[j]],[0,peaks1[i]],'k',color=[0.8,0.8,0.8])
            plt.plot([0,peaks2[j]],[peaks1[i],peaks1[i]],'k',color=[0.8,0.8,0.8])
    if predictions is not None:
        plt.plot(peaks2, predictions, label='Predicted Drift')
    plt.plot([], [], 'bo', markersize=15, label='Confirmed Match')
    plt.plot([], [],'bo', label='Unconfirmed Match')
    plt.plot([], [], 'ro', label='Incorrect Match')
    plt.legend()
    plt.xlim([0,1])
    if not relative:
        plt.ylim([0,1])
    return peaks1, peaks2, predictions


def assess_matches(matches,idx_1,idx_2):
    truth = []
    for i,j in matches:
        if idx_1[i] == idx_2[j]:
            truth.append(1)
        else:
            truth.append(0)
    return truth


def query(data_1_idx,data_2_idx,pos,frag_1):
    idx = data_2_idx[pos]
    for i,ii in enumerate(data_1_idx):
        if ii == idx and frag_1[i] == 1:
            return (i,pos)
    return None


def fit_and_predict(peaks1,peaks2,confirmed_matches,main_K):
    d2_idx = []
    d1_idx = []
    for i,j in confirmed_matches:
        d2_idx.append(j)
        d1_idx.append(i)
    sub_K = main_K[d2_idx,:][:,d2_idx] + np.eye(len(d2_idx))*1e-5
    t = (peaks1[d1_idx] - peaks2[d2_idx])[:,None]
    pred_mu = np.dot(np.dot(main_K[:,d2_idx],np.linalg.inv(sub_K)),t)
    return pred_mu


class SimpleExperiment(object):
    def __init__(self, n_its, dataset1, dataset2, dataset1_idx, dataset2_idx, frag_1, main_K, max_rt=0.02, true_vals=None,
                 true_offset_function=None):
        self.n_its = n_its
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset1_idx = dataset1_idx
        self.dataset2_idx = dataset2_idx
        self.frag_1 = frag_1
        self.main_K = main_K
        self.max_rt = max_rt
        self.true_vals = true_vals
        self.true_offset_function = true_offset_function

    def run(self):
        confirmed_matches = []  # these are the points that will be used for training
        queried_points = []  # store which points we have queried
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matches = closest_match(self.dataset1, self.dataset2, confirmed_matches, queried_points, self.frag_1,
                                    max_rt=self.max_rt, predictions=None)
            truth = assess_matches(matches, self.dataset1_idx, self.dataset2_idx)
            self.plot_figure(matches, confirmed_matches, truth, None)
            for it in range(self.n_its):
                queried_points, confirmed_matches = self.update(queried_points, confirmed_matches)

    def update(self, queried_points, confirmed_matches):
        unqueried = set(range(len(self.dataset1))) - set(queried_points)
        new_match = query(self.dataset1_idx, self.dataset2_idx, np.random.choice(list(unqueried)), self.frag_1)
        if new_match is not None and new_match not in confirmed_matches:
            confirmed_matches.append(new_match)
        pred_mu = fit_and_predict(self.dataset1, self.dataset2, confirmed_matches, self.main_K)
        matches = closest_match(self.dataset1, self.dataset2, confirmed_matches, queried_points, self.frag_1,
                                max_rt=self.max_rt, predictions=pred_mu)
        truth = assess_matches(matches, self.dataset1_idx, self.dataset2_idx)
        self.plot_figure(matches, confirmed_matches, truth, pred_mu)
        return queried_points, confirmed_matches

    def plot_figure(self, matches, confirmed_matches, truth, pred_mu):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.true_vals, -self.true_offset_function, label='True Drift')
        plot_match(self.dataset1, self.dataset2, self.frag_1, matches, confirmed_matches, relative=True,
                   predictions=pred_mu, truth=truth)
        plt.subplot(1, 2, 2)
        plot_datasets(self.dataset1, self.dataset2, self.dataset1_idx, self.dataset2_idx, true_matching=False,
                      matches=matches, confirmed_matches=confirmed_matches)
        plt.show()
