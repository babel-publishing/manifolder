
# TODO decide how to plot multiple snips
        # think that x_ref[1,:] is just
        for snip in range(len(z)):
            if snip == 0:
                x = z[snip][0, :]
                xref1 = x[::manifolder.stepSize]  # downsample, to match the data steps
            else:
                x = z[snip][0, :]
                x = x[::manifolder.stepSize]
                xref1 = np.append(xref1, x)

        print(xref1.shape)
    
        xs = manifolder.Psi[:, 0]
        ys = manifolder.Psi[:, 1]
        zs = manifolder.Psi[:, 2]

        # normalize these to amplitude one?
        print('normalizing amplitudes of Psi in Python ...')
        xs /= np.max(np.abs(xs))
        ys /= np.max(np.abs(ys))
        zs /= np.max(np.abs(zs))

        # xs -= np.mean(xs)
        # ys -= np.mean(ys)
        # zs -= np.mean(zs)

        # xs /= np.std(xs)
        # ys /= np.std(ys)
        # zs /= np.std(zs)

        print(xs.shape)

        lim = 2000
        val = xref1[:lim]
        idx = manifolder.IDX[:lim]
        # print((idx))
        print('Cluster IDs')
        print(type(manifolder.IDX))
        print((manifolder.IDX).shape)
        
        print(len(manifolder.IDX))
        
        plt.figure(figsize=[15, 3])

        plt.plot(xref1[:lim], color='black', label='Timeseries')
        # plt.plot(xs[:lim], linewidth=.5, label='$\psi_0$')
        # plt.plot(ys[:lim], linewidth=.5, label='$\psi_1$')
        # plt.plot(zs[:lim], linewidth=.5, label='$\psi_2$')

        plt.plot(xs[:lim], linewidth=.5, label='psi_0')
        plt.plot(ys[:lim], linewidth=.5, label='psi_1')
        plt.plot(zs[:lim], linewidth=.5, label='psi_2')

        print(np.max(idx))
        plt.plot(idx / np.max(idx) + 1, linewidth=.8, label='IDX')
