from beavermap.generate import BeaverMap 
from typing import Union, Optional
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings 
warnings.simplefilter('ignore')

""" work in progress"""
class BeaverMapPlotter:
    def __init__(
            self,
            BeaverMap:Optional[Union[list,dict,str,BeaverMap]],
            mapping: Optional[Union[np.ndarray,list]]
            ):
        """
        need to add list of dicts
        """
        
        self.mapping = np.array(mapping)

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
        
    def normalise_patches(self,normalise='mean'):
        arr = np.array(self.integrate_arr)

        means = np.array(
            [x[normalise] for x in np.array(self.metadata_arr).flatten()]
            )
        means = means.reshape(self.mapping.shape)
        
        for i,ii in enumerate(arr):
            for j,jj in enumerate(ii):
                for r in range(arr.shape[2]):
                    arr[i][j][r] /= means[i][j]
        return(arr)
    
    def combine_patches(
            self,
            normalise:str,#Optional('str'),
            ):
        if normalise:
            arr = self.normalise_patches(normalise=normalise)
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
            normalise_patches,
            normalise_plots,
            normalise_index_over_all
            ):
        # add more normalisation options 
        if normalise_plots and normalise_plots == 'percentile':
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
    
    def get_colourbar(self,fig,ax,im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.set_yticklabels([])
        cax.set_xticklabels([])
        cax.set_yticks([])
        cax.set_xticks([])
        cax.spines[['top', 'right','bottom','left']].set_visible(False)
        cbar = fig.colorbar(im,ax=cax)
        cbar.ax.set_yticklabels([])
        cbar.ax.set_yticks([])

        return(cbar)
    
    def plot_combined(self,
                      index:int = None,#Optional(int) = None,
                      normalise_patches:str = 'median',#Optional(str) = None,
                      normalise_plots:str = 'percentile',#Optional(str) = 'percentile',
                      normalise_index_over_all:bool = True,
                      cmap:str ='magma',
                      figsize:tuple =(10,10),
                      nrows:int= 7,
                      gridspec_kw = {'wspace':0.1, 'hspace':0.1},
                      scale_bar:bool = False,
                      scale_bar_kws:dict = {'size':50,'label':'50 $\\mu$m','loc': 'lower center','pad':0.1,'color':'tab:grey','frameon':False,'size_vertical':1},
                      colourbar:bool = True,
                      annotate = True,
                      **kws):
        
        combined_patches = self.combine_patches(normalise=normalise_patches)

        norm = self.get_matplotlib_norm(
            index,
            combined_patches,
            normalise_patches,
            normalise_plots,
            normalise_index_over_all
            )
            # add more 
        ncols,_nrows = (int(np.round(combined_patches.shape[0]/nrows)),nrows)
        
        if not index: 
            fig, axes = plt.subplots(ncols=ncols,nrows=_nrows,figsize=figsize,gridspec_kw=gridspec_kw)

            for i,patch in enumerate(combined_patches):
                im= axes.flatten()[i].imshow(patch,norm=norm,cmap=cmap)  # Example plot
                if colourbar:
                    cbar = self.get_colourbar(ax=axes.flatten()[i],fig=fig,im=im)

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

            im = axes.imshow(combined_patches[index],norm=norm,cmap=cmap)
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_yticks([])
            axes.set_xticks([])
            if scale_bar:
                scalebar = AnchoredSizeBar(axes.transData,**scale_bar_kws)
                axes.add_artist(scalebar)

            if colourbar:
                cbar = self.get_colourbar(ax=axes,fig=fig,im=im)

            if annotate:
                text = self.two_theta_regions[index]
                axes.set_ylabel(text)

        return(fig,axes)