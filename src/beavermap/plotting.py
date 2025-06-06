from beavermap.generate import BeaverMap 
from typing import Union, Optional
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib 
import warnings 
warnings.simplefilter('ignore')

""" work in progress"""
class BeaverMapPlotter:
    def __init__(
            self,
            BeaverMap:Optional[Union[list,dict,str,BeaverMap]],
            mapping: Optional[Union[np.ndarray,list]],
            normalise_patches_method:Optional[Union['median','mean',None]],
            ):
        """
        need to add list of dicts
        """
        
        self.mapping = np.array(mapping)
        self.normalise_patches_method = normalise_patches_method

        if isinstance(BeaverMap,list):
            if not isinstance(BeaverMap[0],dict):
                if not isinstance(BeaverMap[0],str):
                    #list of beavermap objects that is not a path or dict
                    self.max_sum_arr = np.zeros(self.mapping.shape).tolist()
                    self.integrate_arr = np.zeros(self.mapping.shape).tolist()
                    self.metadata_arr = np.zeros(self.mapping.shape).tolist()
                    for i,item in enumerate(BeaverMap):
                        pos = np.where(self.mapping==i)
                        if hasattr(item, 'max_sum_results'):
                            if len(pos) == 2:
                                self.max_sum_arr[pos[0][0]][pos[1][0]] = item.max_sum_results
                            else:
                                self.max_sum_arr[pos[0][0]] = item.max_sum_results
                        if hasattr(item,'integrate_results'):
                            if len(pos) == 2:
                                self.integrate_arr[pos[0][0]][pos[1][0]] = item.integrate_results
                                self.metadata_arr[pos[0][0]][pos[1][0]] = item.__dict__
                            else:
                                self.integrate_arr[pos[0][0]] = item.integrate_results
                                self.metadata_arr[pos[0][0]] = item.__dict__
                else:
                    #list of beavermap strings (paths to .npz files) that is not a Beavermap object or dict
                    self.max_sum_arr = None #np.zeros(self.mapping.shape).tolist() # currently not doing max_sum_arr for these files
                    self.integrate_arr = np.zeros(self.mapping.shape).tolist()
                    self.metadata_arr = np.zeros(self.mapping.shape).tolist()
                    for i,item in enumerate(BeaverMap):
                        pos = np.where(self.mapping==i)
                        data = np.load(item,allow_pickle=True)
                        _arr = data['features']
                        _metadata = data['metadata'].item()
                        if len(pos) == 2:
                            self.integrate_arr[pos[0][0]][pos[1][0]] = _arr
                            self.metadata_arr[pos[0][0]][pos[1][0]] = _metadata
                        else:
                            self.integrate_arr[pos[0][0]] = _arr
                            self.metadata_arr[pos[0][0]] = _metadata

        elif isinstance(BeaverMap,dict):
            if 'max_sum_results' in BeaverMap:
                self.max_sum_arr = [BeaverMap['max_sum_results']]
            if 'integrate_results' in BeaverMap:
                self.integrate_arr = [BeaverMap['integrate_results']]
            self.metadata_arr = [BeaverMap]

        else:
            if hasattr(BeaverMap, 'max_sum_results'):
                self.max_sum_arr = [BeaverMap.max_sum_results]
            if hasattr(BeaverMap,'integrate_results'):
                self.integrate_arr = [BeaverMap.integrate_results]

            self.metadata_arr = [BeaverMap.__dict__]

        self.check_two_theta_regions()

    def check_two_theta_regions(self):

        two_theta_regions = []
        for data in np.array(self.metadata_arr).flatten():
            two_theta_regions.append(data['regions'])

        if all([x == two_theta_regions[0] for x in two_theta_regions]):
            self.two_theta_regions = two_theta_regions[0]
        else:
            raise AttributeError('not all data have the same two theta regions!')
        
    def normalise_patches_by(self):
        arr = np.array(self.integrate_arr)

        means = np.array(
            [x[self.normalise_patches_method] for x in np.array(self.metadata_arr).flatten()]
            )
        means = means.reshape(self.mapping.shape)
        
        for i,ii in enumerate(arr):
            for j,jj in enumerate(ii):
                for r in range(arr.shape[2]):
                    arr[i][j][r] /= means[i][j]
        return(arr)
    
    def combine_patches(
            self,
            ):
        if self.normalise_patches_method:
            arr = self.normalise_patches_by()
        else:
            arr = np.array(self.integrate_arr)
        combined_data = []
        for i in range(arr.shape[2]):
            _arr = arr[:,:,i,:]
            t1 = np.hstack(_arr)
            combined_data.append(np.hstack(t1))
    
        return(np.array(combined_data))
    
    def get_matplotlib_norm(
            self,
            index,
            patches,
            normalise_plots_by,
            normalise_index_over_all
            ):
        # add more normalisation options 
        if normalise_plots_by and normalise_plots_by == 'percentile':
            if not index:
                # normalise over all the patches
                norm = Normalize(
                    vmin=np.percentile(patches,10),vmax=np.percentile(patches,90)
                    )
            else:
                if not normalise_index_over_all:
                    norm = Normalize(
                        vmin=np.percentile(patches[index],10),
                        vmax=np.percentile(patches[index],90)
                    )
                else:
                    norm = Normalize(
                        vmin=np.percentile(patches,10),vmax=np.percentile(patches,90)
                        )
        return(norm)
    
    def add_colourbar(
            self,
            fig,
            ax,
            im,
            show_axes:bool=False):
        """
        generates a colourbar 
        Args:
        fig: maptlotlib figure 
        ax: matplotlib.axes.Axes
        im: matplotlib.imshow
        show_axes:bool = False ; show ticks and ticklabels 
        Returns:
        matplotlib.colorbar 
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.set_yticklabels([])
        cax.set_xticklabels([])
        cax.set_yticks([])
        cax.set_xticks([])
        cax.spines[['top', 'right','bottom','left']].set_visible(False)
        cbar = fig.colorbar(im,ax=cax)
        if not show_axes:
            cbar.ax.set_yticklabels([])
            cbar.ax.set_yticks([])
        return(cbar)
    
    def get_subplot_grid(
            self,
            length:int,
            ncols:Optional[int],
            nrows:Optional[int]
            ):
        
        if not any([ncols,nrows]):
            nrows = int(np.ceil(np.sqrt(length)))
            ncols = nrows 
        elif not ncols: 
            ncols = int(np.ceil(length/nrows))
        elif not nrows: 
            nrows = int(np.ceil(length/ncols))
        else:
            if not ncols * nrows > length:
                raise ValueError(f'ncols  and nrows ({ncols} * {nrows} = {ncols * nrows}) < length ({length})')
            
        return(ncols,nrows)

    def plot_combined(
        self,
        nrows: Optional[int] = None ,
        ncols: Optional[int] = None ,
        index: Optional[int] = None,  
        normalise_plots_method: Optional[Union['percentile','mean','median',None]] = 'percentile', 
        normalise_index_over_all: bool = True,
        cmap: str = "magma",
        figsize: tuple = (10, 10),
        scale_bar: bool = False,
        colourbar: bool = True,
        annotate: bool = True,
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        scale_bar_kws: dict = {
            "size": 50,
            "label": "50 $\\mu$m",
            "loc": "lower center",
            "pad": 0.1,
            "color": "tab:grey",
            "frameon": False,
            "size_vertical": 1,
        },
        aspect='auto'
                ):
        
        combined_patches = self.combine_patches()

        norm = self.get_matplotlib_norm(
            index,
            combined_patches,
            normalise_plots_method,
            normalise_index_over_all
            )

        
        if not index: 
            #define nrows and ncols - defaults to a squareish grid, but can specify nrows or ncols if needed
            ncols,nrows = self.get_subplot_grid(length = len(self.two_theta_regions),ncols=ncols,nrows=nrows)

            fig, axes = plt.subplots(
                ncols=ncols,nrows=nrows,figsize=figsize,gridspec_kw=gridspec_kw
                )

            for i,patch in enumerate(combined_patches):
                im= axes.flatten()[i].imshow(patch,norm=norm,cmap=cmap,aspect=aspect)  # Example plot
                if colourbar:
                    cbar = self.add_colourbar(ax=axes.flatten()[i],fig=fig,im=im)

                axes.flatten()[i]
                axes.flatten()[i].set_yticklabels([])
                axes.flatten()[i].set_xticklabels([])
                axes.flatten()[i].set_yticks([])
                axes.flatten()[i].set_xticks([])
                if scale_bar:
                    scalebar = AnchoredSizeBar(axes.flatten()[i].transData, **scale_bar_kws)
                    axes.flatten()[i].add_artist(scalebar)
                
                if annotate:
                    text = self.two_theta_regions[i]
                    axes.flatten()[i].set_ylabel(text)
                    axes.flatten()[i].set_xlabel(i)

            # Remove any unused subplots
            for j in range(combined_patches.shape[0], len(axes.flatten())):
                fig.delaxes(axes.flatten()[j])

            
        else:
            fig,axes = plt.subplots(figsize=figsize,gridspec_kw=gridspec_kw)

            im = axes.imshow(combined_patches[index],norm=norm,cmap=cmap,aspect=aspect)
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_yticks([])
            axes.set_xticks([])
            if scale_bar:
                scalebar = AnchoredSizeBar(axes.transData,**scale_bar_kws)
                axes.add_artist(scalebar)

            if colourbar:
                cbar = self.add_colourbar(ax=axes,fig=fig,im=im)

            if annotate:
                text = self.two_theta_regions[index]
                axes.set_ylabel(text)

        return(fig,axes)