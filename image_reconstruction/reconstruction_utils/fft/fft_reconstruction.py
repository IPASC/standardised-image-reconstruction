"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Jenni Poimala
SPDX-FileCopyrightText: 2021 Andreas Hauptmann
SPDX-License-Identifier: MIT

"""

import torch
import numpy as np
from scipy.signal import hilbert


from scipy.interpolate import RegularGridInterpolator
import math 

import matplotlib.pyplot as plt

import copy

def fft_reconstruction(time_series_data, detection_elements, sampling_rate, field_of_view, spacing_m, speed_of_sound_in_m_per_s):
    """

    Parameters
    ----------
    time_series_data
    detection_elements
    field_of_view
    spacing_m
    speed_of_sound_in_m_per_s

    Returns
    -------

    """


    time_spacing_in_s = 1.0 / sampling_rate
    
    positions = detection_elements["positions"]

    
    
    if field_of_view is None:
        
        
        # find the axis of the planar sensor 
        sensor_axis=[i for i, e in enumerate([len(set(positions[:, 0])), len(set(positions[:, 1])), len(set(positions[:, 2]))])  if e != 1]
    
        
        if len(sensor_axis)>1:
            
            print('ERROR: Sensor is not a planar sensor')
        
         
        # make grid for the sensor axis 
        x=np.arange(positions[:, sensor_axis[0]].min(), positions[:, sensor_axis[0]].max()+spacing_m , spacing_m)
              
         
        # sensor locations
        x_sensor=positions[:,sensor_axis[0]]
              
            
        # locate sensors to the sensor axis grid that is find nearest pixel   
        idxs=find_nearest_index(x, x_sensor)
            
              
        
        # create a new zero pressure data set       
        pT=np.zeros((x.shape[0],time_series_data.shape[1]) )
              
              
              
        # put the measured data into correct location      
        pT[idxs]=time_series_data
              
       
            
        # Use all data points      
        NtFactor=1
              
                   
        # spacing      
        dx=spacing_m
        dy=dx
        
        # speed of sound
        c=speed_of_sound_in_m_per_s
        
        # time spacing
        dt=time_spacing_in_s
        
        
        
        # FFT based reconstruction 
        
        
        # mirror the time domain data 
        pT=np.transpose(pT)     
         
        pUp= np.flipud(pT)
        p0 = np.concatenate((pUp,pT[1::,:]),0)
        
        size=p0.shape
        
        Nt=size[0]
        Ny=size[1]
        
       
        output = np.zeros((Ny, 1, math.floor(Nt/NtFactor/2)))
       
        # compute kgrids
        kgridBack = kgrid2D(math.ceil(Nt/NtFactor), dx, Ny, dy, c)
        
        kgridBackC = kgrid2D(Nt, dx, Ny, dy,  dx/dt)
     
        
     
        # computational grid for w and w_new
        c=kgridBackC.c
        w=c*kgridBackC.kx
        w_new = kgridBack.c*kgridBack.k
        
        
        # scalig factor
        sf=np.square(w/c) - np.square(kgridBackC.ky)
        sf=c*c*np.sqrt( sf.astype(np.complex)  )
        sf=np.divide(sf , 2*w)

        
        idx1  = np.where( (w == 0) & (kgridBackC.ky==0) )
        sf[idx1]=c/2 
        
         
        idx2  = np.where(np.abs(w)< np.abs(c*kgridBackC.ky) )
        sf[idx2]=0 
        
    
        idx  = np.where(np.isnan(sf))
        sf[idx]=0   
        
     
        # computational grids needed for interpolation
        ky=kgridBackC.ky_vec
        kyI=kgridBack.ky
    
        
        w=c*kgridBackC.kx_vec
        wI=w_new
        

        points_inv   = (w.flatten(), ky.flatten())
        pointsI_inv   = (wI.flatten() ,kyI.flatten())

   
        NxyzI  = wI.shape

        # indexes for correct part of reconstruction
        indS=math.ceil((p0.shape[0]+1)/2/NtFactor)
        indE=math.floor((p0.shape[0])/NtFactor)+1
       
        # the FFT of the input data and scaling
        p0 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(p0)))
        p0 = p0*sf
   
      
        # interpolation   
        gi = RegularGridInterpolator(points_inv, p0, bounds_error=False, fill_value=0)
       
        p0=gi(pointsI_inv)
       
       
        p0=np.reshape(p0, (NxyzI[0], NxyzI[1]))

     
        # the inverse FFT
        p0 = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(p0)))
        
        
        # take correct part of reconstruction and scale
        p0=2*2*p0.real[indS:indE, :]/c /NtFactor 
        
        
        # shape result to correct form
        p0=np.transpose(p0) 
        
        

        output[:, :, :]=p0[:, None, :]
        
        
    else:
        
        
            
        field_of_view_voxels = np.round(field_of_view / spacing_m).astype(int)


        x_dim = (field_of_view_voxels[1] - field_of_view_voxels[0])
        y_dim = (field_of_view_voxels[3] - field_of_view_voxels[2])
        z_dim = (field_of_view_voxels[5] - field_of_view_voxels[4])

        # Just to make sure not to allocate a 0-dimensional array
        if x_dim < 1:
            x_dim = 1
        if y_dim < 1:
            y_dim = 1
        if z_dim < 1:
            z_dim = 1
    
    
        dim=[x_dim, y_dim, z_dim]
        
        # find the axis of the planar sensor 
        sensor_axis=[i for i, e in enumerate([len(set(positions[:, 0])), len(set(positions[:, 1])), len(set(positions[:, 2]))])  if e != 1]
   
       
        if len(sensor_axis)>1:
           
            print('ERROR: Sensor is not a planar sensor')
       
        
        # make grid for the sensor axis 
        x=np.arange(field_of_view[2*sensor_axis[0]],  field_of_view[2*sensor_axis[0]+1], spacing_m)
                 
        
        # sensor locations
        x_sensor=positions[:,sensor_axis[0]]
             
           
        # locate sensors to the sensor axis grid that is find nearest pixel   
        idxs=find_nearest_index(x, x_sensor)
           
             
       
        # create a new zero pressure data set       
        pT=np.zeros((x.shape[0],time_series_data.shape[1]) )
             
             
             
        # put the measured data into correct location      
        pT[idxs]=time_series_data
        
        
        # determine NtFactor = relation between number of pixels in depht direction and number of time points  
        
        dim_am=copy.deepcopy(dim)
        
        dim_am.remove(dim_am[sensor_axis[0]])
        
       
        # Use only part of data points
        NtFactor=math.floor(time_series_data.shape[1]/max(dim_am))
        
                     
        # spacing
        dx=spacing_m
        dy=dx
        
        # speed of sound
        c=speed_of_sound_in_m_per_s
        
        # time spacing
        dt=time_spacing_in_s
        
        
        
        # FFT based reconctruction
        
        
        # mirror the time domain data 
        pT=np.transpose(pT)     
         
        pUp= np.flipud(pT)
        p0 = np.concatenate((pUp,pT[1::,:]),0)
        
        size=p0.shape
        
        Nt=size[0]
        Ny=size[1]
        
        output = np.zeros((dim))
    
       
        # compute kgrids
        kgridBack = kgrid2D(math.ceil(Nt/NtFactor), dx, Ny, dy, c)
        
        kgridBackC = kgrid2D(Nt, dx, Ny, dy,  dx/dt)
        
     
        # computational grid for w and w_new
        c=kgridBackC.c
        w=c*kgridBackC.kx
        w_new = kgridBack.c*kgridBack.k
        
        
        # scalig factor
        sf=np.square(w/c) - np.square(kgridBackC.ky)
        sf=c*c*np.sqrt( sf.astype(np.complex)  )

        sf=np.divide(sf , 2*w)

        idx1  = np.where( (w == 0) & (kgridBackC.ky==0) )
        sf[idx1]=c/2      
         
        idx2  = np.where(np.abs(w)< np.abs(c*kgridBackC.ky) )
        sf[idx2]=0 
              
        idx  = np.where(np.isnan(sf))
        sf[idx]=0   
        
        
        
        # computational grids needed for interpolation
        ky=kgridBackC.ky_vec
        kyI=kgridBack.ky
    
        
        w=c*kgridBackC.kx_vec
        wI=w_new
        

        points_inv   = (w.flatten(), ky.flatten())
        pointsI_inv   = (wI.flatten() ,kyI.flatten())

   
        NxyzI  = wI.shape

        
        # indexes for correct part of reconstruction
        indS=math.ceil((p0.shape[0]+1)/2/NtFactor)
        indE=math.floor((p0.shape[0])/NtFactor)+1
       
       
        # the FFT of the input data and scaling
        p0 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(p0)))
        p0 = p0*sf
   
        # interpolation
        gi = RegularGridInterpolator(points_inv, p0, bounds_error=False, fill_value=0)
       
        p0=gi(pointsI_inv)
       
    
        p0=np.reshape(p0, (NxyzI[0], NxyzI[1]))

     
        # the inverse FFT
        p0 = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(p0)))
        
        
        # take correct part of reconstruction and scale
        p0=2*2*p0.real[indS:indE, :]/c /NtFactor 

        p0=np.transpose(p0) 
        
        
        # take only part that corrensponds to field of view
        p0=p0[0:dim[0], 0:dim[2]]
        
 
        # shape result to correct form
        output[:, :, :]=np.reshape(p0, (dim))

    return output



 
class kgrid2D(object):
    def __init__(self, Nx, dx, Ny, dy,  c):

        [k, kx, ky, kx_vec, ky_vec ] = makeKgrid2D(Nx, dx, Ny, dy)    

        self._kx=kx
        self._ky=ky
        self._k=k
        
        self._kx_vec=kx_vec
        self._ky_vec=ky_vec

        

        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.c  = c

    @property
    def kx(self):        
         return self._kx
     
        
    @property
    def ky(self):        
         return self._ky
         
     
    @property
    def k(self):        
         return self._k
     
        
    @property
    def kx_vec(self):        
         return self._kx_vec
     
    @property
    def ky_vec(self):        
         return self._ky_vec
        
     
        
     

       
def makeKgrid2D(Nx, dx, Ny, dy):
    
    kx_vec=makeDim(Nx, dx)
    ky_vec=makeDim(Ny, dy)


    kx = np.tile(kx_vec.reshape(-1,1), (1, Ny))
    

    ky=np.tile(ky_vec, (Nx, 1))
    
    
    k=np.zeros([Nx, Ny])
    
    k=(kx_vec**2).reshape(-1,1)+k


    k=(ky_vec**2)+k
    
    

    k=np.sqrt(k)
                   
    return k, kx, ky, kx_vec, ky_vec 


      
def makeDim(Nx, dx):
    
    
    if  (Nx % 2) == 0:
    
        
        nx = np.arange(-Nx/2, Nx/2)/Nx
    
    else:
        
        
        nx = np.arange(-(Nx-1)/2, (Nx)/2)/Nx
        
    # force middle value to be zero in case 1/Nx is a recurring number and the series doesn't give exactly zero
    nx[math.floor(Nx/2)] = 0
            
    # define the wavenumber vector components
    kx_vec = (2*np.pi/dx)*nx
    
    return kx_vec


def find_nearest_index(array, value):
    
    idxs=[]
    
    for ii in value:
        
        array = np.asarray(array)
    
        idx = (np.abs(array - ii)).argmin()
        
        idxs.append(idx)
    
    
    return idxs