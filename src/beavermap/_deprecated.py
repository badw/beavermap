    def integrate_worker(
            self,
            in_q,
            out_q,
            args,
            regions,
    ):
        while True:
            ### queue memory checker here?
            image = in_q.get()

            i0 = int(np.floor(image / self.dim1))  # check these...
            i1 = image - self.dim1 * int(np.floor(image / self.dim1))

            with h5py.File(self.h5_file, "r") as f:
                data = f[self.location][image]
                #integrated = np.array(
                #    self.ai.integrate1d(
                #        data=f[self.location][image], mask=self.mask_data, **args
                #    )[0:2]
                #)
            #integrated = np.array(
            #    self.ai.integrate1d(data=data,mask=self.mask_data,**args),copy=False
            #    )[0:2]
            integrated = np.asarray(self.integrate_function(data=data,mask=self.mask_data,**args))[0:2]

            full_data = np.zeros((len(regions), self.dim0, self.dim1))

            for i, r in enumerate(regions):
                _arrmask = (integrated[0] >= r[0]) & (integrated[0] <= r[1])
                full_data[i][i0, i1] = np.sum(integrated[1][_arrmask])

            del integrated 

            out_q.put(full_data)

    def integrate(
            self, 
            integrate_args=None, 
            regions=[[0, 100]], 
            ):
        
        if not integrate_args:
            self.default_integrate_args
        else:
            self.integrate_args = integrate_args

        final_results = []

        ctx = pmp.get_context("fork")

        self.in_queue = ctx.Queue()
        self.out_queue = ctx.Queue()

        image_range = np.arange(self.n_images)

        for image in image_range:
            self.in_queue.put(image)

        self.workers = []
        for _ in range(self.nworkers):
            self.workers.append(
                ctx.Process(
                    target=self.integrate_worker,
                    args=(self.in_queue, self.out_queue, self.integrate_args, regions),
                )
            )
        for w in self.workers:
            w.start()

        _bar_format = "{desc} {n_fmt}/{total_fmt}|{percentage:3.0f}%|{bar}| {elapsed}<    {remaining}{postfix}"
        total = int(len(image_range) * len(image_range) / 2 - len(image_range) / 2)
        divider = len(image_range) / total

        with tqdm.tqdm(
            total=len(image_range),
            desc=f"performing integration",
            bar_format=_bar_format,
            ncols=80,
        ) as pbar:
            results = []
            for ii, image in enumerate(image_range):
                results.append(self.get_result_from_queue(pbar, np.round(divider * ii)))

        self.terminate_workers()

        final_results.extend([np.array(results).sum(axis=0)])

        self.terminate_workers()

        return np.array(final_results).sum(axis=0)