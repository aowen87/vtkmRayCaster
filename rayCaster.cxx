/***
* @author: Alister Maguire
*
* This is a vtk-m version of a basic ray casting
* algorithm. NOTES: 
* 1. It is currently set up
*    for one specific input (the camera position and
*    transfer function are specific to this data set). 
* 2. I'm using a binary search to search for points
*    along the ray, which is not scalable. 
* 3. The perspective is slightly warped, because I
*    am using a lazy method for determining the start
*    and stop positions of the rays being cast. (it's 
*    not visually noticable with this data set, so it's
*    not a huge deal) 
***/
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <vector>

#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/exec/ExecutionObjectBase.h>
#include <vtkm/VectorAnalysis.h>
#include <vtkm/Math.h>
#include <vtkm/cont/DynamicArrayHandle.h>
#include <vtkm/cont/internal/ArrayPortalFromIterators.h>
#include <vtkm/rendering/CanvasRayTracer.h>

using std::cerr;
using std::cout;
using std::endl;

#define TIMING true
#define NUM_SAMPLES 500
#define BASE_SAMP_RATE 200
#define PI 3.14159
#define H 1000
#define W 1000


namespace vtkm {
namespace exec {
    
struct TransferFunction : public vtkm::exec::ExecutionObjectBase
{
    vtkm::Float32   min;
    vtkm::Float32   max;
    vtkm::Int32     numBins;
    unsigned char  colors[3*256];  // size is 3*numBins
    vtkm::Float32  opacities[256]; // size is numBins

    VTKM_EXEC
    vtkm::Int32 GetBin(const vtkm::Float32 value) const 
    { return floor((value - min)/((max - min)/numBins)); }

    template <typename OpType,
              typename RGBType> 
    VTKM_EXEC
    void ApplyTransferFunction(const vtkm::Float32 &value, 
                                     RGBType       &RGB, 
                                     OpType        &opacity) const
    {
        if (value < min || value > max)
        {
            RGB[0] = 0; 
            RGB[1] = 0;
            RGB[2] = 0;
            opacity = 0.0;
            return;
        }
        vtkm::Int32 bin = GetBin(value);
        RGB[0]  = colors[3*bin+0];
        RGB[1]  = colors[3*bin+1];
        RGB[2]  = colors[3*bin+2];
        opacity = 1 - pow((1- opacities[bin]), vtkm::Float32(BASE_SAMP_RATE)/vtkm::Float32(NUM_SAMPLES));
    }

    VTKM_CONT
    TransferFunction()
    {

        vtkm::Int32  i;

        min = 10;
        max = 15;
        numBins = 256;
        unsigned char charOpacity[256] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14, 14, 
            14, 14, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 5, 4, 3, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 16, 16, 15, 14, 13, 12, 11, 9, 8, 
            7, 6, 5, 5, 4, 3, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 18, 20, 22, 24, 27, 29, 32, 
            35, 38, 41, 44, 47, 50, 52, 55, 58, 60, 62, 64, 66, 67, 68, 69, 70, 70, 70, 69, 68, 67, 
            66, 64, 62, 60, 58, 55, 52, 50, 47, 44, 41, 38, 35, 32, 29, 27, 24, 22, 20, 20, 23, 28, 
            33, 38, 45, 51, 59, 67, 76, 85, 95, 105, 116, 127, 138, 149, 160, 170, 180, 189, 198, 
            205, 212, 217, 221, 223, 224, 224, 222, 219, 214, 208, 201, 193, 184, 174, 164, 153, 
            142, 131, 120, 109, 99, 89, 79, 70, 62, 54, 47, 40, 35, 30, 25, 21, 17, 14, 12, 10, 8, 
            6, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        for (i = 0 ; i < 256 ; i++)
            opacities[i] = charOpacity[i]/255.0;
        const vtkm::Int32 numControlPoints = 8;
        unsigned char controlPointColors[numControlPoints*3] = { 
               71, 71, 219, 0, 0, 91, 0, 255, 255, 0, 127, 0, 
               255, 255, 0, 255, 96, 0, 107, 0, 0, 224, 76, 76 
           };
        double controlPointPositions[numControlPoints] = { 0, 0.143, 0.285, 0.429, 0.571, 0.714, 0.857, 1.0 };
        for (i = 0 ; i < numControlPoints-1 ; i++)
        {
            vtkm::Int32 start = controlPointPositions[i]*numBins;
            vtkm::Int32 end   = controlPointPositions[i+1]*numBins+1;
            if (end >= numBins)
                end = numBins-1;
            for (vtkm::Int32 j = start ; j <= end ; j++)
            {
                double proportion = (j/(numBins-1.0)-controlPointPositions[i])/
                                    (controlPointPositions[i+1]-controlPointPositions[i]);
                if (proportion < 0 || proportion > 1.)
                    continue;
                for (vtkm::Int32 k = 0 ; k < 3 ; k++)
                    colors[3*j+k] = proportion*(controlPointColors[3*(i+1)+k]-controlPointColors[3*i+k])
                                     + controlPointColors[3*i+k];
            }
        }    
    }
     
};//Transfer function


class Ray : public vtkm::exec::ExecutionObjectBase
{

  private:
    vtkm::Vec<vtkm::Float32, 3> ray;  
    std::vector<vtkm::Float32> samples;

  public:
    VTKM_EXEC
    Ray() : samples(NUM_SAMPLES, 0) {}

    VTKM_EXEC
    std::vector<vtkm::Float32> GetSamples() { return samples; }

    VTKM_EXEC
    vtkm::cont::ArrayHandle<vtkm::Float32> GetSampleHandle()
    {
        vtkm::cont::ArrayHandle<vtkm::Float32> handle =
            vtkm::cont::make_ArrayHandle(samples);
        return handle;
    }

    VTKM_EXEC
    vtkm::Vec<vtkm::Float32, 3> GetRay() { return ray; }

    VTKM_EXEC
    void SetRay(vtkm::Vec<vtkm::Float32, 3> inRay)
    {
        ray = inRay;
    }

};//Ray


struct Camera : public vtkm::exec::ExecutionObjectBase
{

  public:
    vtkm::Float32               near, far;
    vtkm::Float32               angle;
    vtkm::Vec<vtkm::Float32, 3> position;
    vtkm::Vec<vtkm::Float32, 3> focus;
    vtkm::Vec<vtkm::Float32, 3> up;
    vtkm::Vec<vtkm::Float32, 3> look;

    VTKM_CONT
    void InitCamera()
    {
        focus    = vtkm::make_Vec(0.0, 0.0, 0.0);
        up       = vtkm::make_Vec(0.0, 1.0, 0.0); 
        position = vtkm::make_Vec(-8.25e+7, -3.45e+7, 3.35e+7);
        angle    = 30.0;
        near     = 7.5e+7;
        far      = 1.4e+8;
        CalculateLook();
    }

    VTKM_CONT 
    void CalculateLook()
    {
        look = focus - position;
        vtkm::Normalize(look);
    }

};//Camera


struct Screen : public vtkm::exec::ExecutionObjectBase
{

  public:
    vtkm::Int32                              height;
    vtkm::Int32                              width;
    std::vector<vtkm::Vec<unsigned char, 3>> pBuffer;

    VTKM_CONT
    void SetDimensions(vtkm::UInt32 w, vtkm::UInt32 h) 
    {
        width  = w;
        height = h;
    }

    VTKM_CONT
    void Initialize()
    {
        vtkm::UInt32 npixels = width*height;
        for (int i = 0; i < npixels; ++i)
           pBuffer.push_back(vtkm::make_Vec(0, 0, 0));
    }

    
};//Screen
}//exec namespace


namespace worklet {

class RayCaster : public vtkm::worklet::WorkletMapField
{

  protected:

    typedef vtkm::cont::ArrayHandle<vtkm::FloatDefault> DefaultHandle;
    typedef vtkm::cont::ArrayHandleCartesianProduct<DefaultHandle, DefaultHandle, DefaultHandle>
      CartesianArrayHandle;
    typedef typename DefaultHandle::ExecutionTypes<vtkm::cont::DeviceAdapterTagCuda>::PortalConst DefaultConstHandle;
    typedef typename CartesianArrayHandle::ExecutionTypes<vtkm::cont::DeviceAdapterTagCuda>::PortalConst CartesianConstPortal;


    DefaultConstHandle   field;
    DefaultConstHandle   coordPortals[3];
    CartesianConstPortal coordinates;
    vtkm::Vec<Int32, 3>  dims;
       

  public:
    RayCaster (const CartesianArrayHandle &coords, const DefaultHandle &inField) : 
               coordinates(coords.PrepareForInput(vtkm::cont::DeviceAdapterTagCuda())), 
               field(inField.PrepareForInput(vtkm::cont::DeviceAdapterTagCuda()))
    {
        coordPortals[0] = coordinates.GetFirstPortal();
        coordPortals[1] = coordinates.GetSecondPortal();
        coordPortals[2] = coordinates.GetThirdPortal();
        dims = vtkm::make_Vec(coordPortals[0].GetNumberOfValues(),
                              coordPortals[1].GetNumberOfValues(),
                              coordPortals[2].GetNumberOfValues());
    };

    typedef void ControlSignature(FieldOut<>     Pixels,
                                  ExecObject     Camera,
                                  ExecObject     Screen,
                                  ExecObject     TransferFunction);

    typedef void ExecutionSignature(_1,
                                    _2,
                                    _3,
                                    _4,
                                    InputIndex);
    typedef _1 InputDomain;

    template<typename CameraType,
             typename ScreenType,
             typename PixelType,
             typename TransferFunctionType>
    VTKM_EXEC
    void operator() (PixelType                  &pixel,
                     const CameraType           &camera,
                     const ScreenType           &screen,
                     const TransferFunctionType &tf,
                     const vtkm::Id             &idx) const
    {
        //Get the coordinates associated with this index
        vtkm::Float32 xCoord = idx%screen.width;
        vtkm::Float32 yCoord = (idx/screen.width)%screen.height;

        //calculate the ray for this pixel
        vtkm::Float32 width  = screen.width;
        vtkm::Float32 height = screen.height;

        vtkm::Vec<vtkm::Float32, 3> ru;
        vtkm::Vec<vtkm::Float32, 3> rv;
        vtkm::Vec<vtkm::Float32, 3> rx;
        vtkm::Vec<vtkm::Float32, 3> ry;

        ru = vtkm::Cross(camera.look, camera.up);
        vtkm::Normalize(ru);

        rv = vtkm::Cross(camera.look, ru);
        vtkm::Normalize(rv);
        
        vtkm::Float32 xAlpha = (2.0*vtkm::Tan((camera.angle/2.0))/PI)/width;
        vtkm::Float32 yAlpha = (2.0*vtkm::Tan((camera.angle/2.0))/PI)/height;

        rx = xAlpha*ru;
        ry = yAlpha*rv;

        vtkm::Vec<vtkm::Float32, 3> ray = 
            (camera.look + ((2.0*xCoord + 1.0 - width) / 2.0)*rx +
                    ((2.0*yCoord + 1.0 - height) / 2.0)*ry);

        vtkm::Normalize(ray);

        //Retrieve samples
        vtkm::Vec<vtkm::Float32, 3> curPos(0);
        curPos = camera.position;

        vtkm::Float32 stepSize = (camera.far - camera.near)/NUM_SAMPLES;
        curPos += ray*camera.near;

        vtkm::Vec<vtkm::Float32, NUM_SAMPLES> samples(0.0);
        vtkm::Int32 i = 0; 

        while (i < NUM_SAMPLES)
        {
            vtkm::Int32 ptIdx[3];

            FindGridPoint(curPos, ptIdx);
            vtkm::Int32 cellId = ptIdx[2]*(dims[0]-1)*(dims[1]-1)
                                +ptIdx[1]*(dims[0]-1)+ptIdx[0];

            vtkm::Float32 bbox[6];
            BoundingBoxForCell(cellId, bbox);
             
            if (bbox[0] == -1.0)
            {
               samples[i] = 0.0;
               curPos    += stepSize*ray;
               ++i; 
               continue;
            }

            samples[i] = GetCellSample(curPos, bbox, ptIdx);
            curPos    += stepSize*ray;
            ++i;
        }

        //Composite samples
        unsigned char frontRGB[] = {0, 0, 0};
        vtkm::Float32 frontOp = 0.0;
             
        for (vtkm::Int32 i = 0; i < NUM_SAMPLES; ++i)
        {
            unsigned char sampleRGB[3];
            vtkm::Float32 sampleOp = 0.0;
            tf.ApplyTransferFunction(samples[i], sampleRGB, sampleOp);
            
            if (samples[i] >= tf.min && samples[i] <= tf.max)
            {
                vtkm::Int32 bin = tf.GetBin(samples[i]);
                sampleRGB[0]  = tf.colors[3*bin+0];
                sampleRGB[1]  = tf.colors[3*bin+1];
                sampleRGB[2]  = tf.colors[3*bin+2];
                sampleOp = 1 - pow((1- tf.opacities[bin]), vtkm::Float32(BASE_SAMP_RATE)/vtkm::Float32(NUM_SAMPLES));
            }
            else
            {
                sampleRGB[0] = 0; 
                sampleRGB[1] = 0;
                sampleRGB[2] = 0;
                sampleOp = 0.0;
            }
               
            vtkm::Float32 fDiff = (1.0 - frontOp);
            frontRGB[0]  = frontRGB[0] + fDiff*sampleOp*sampleRGB[0];
            frontRGB[1]  = frontRGB[1] + fDiff*sampleOp*sampleRGB[1];
            frontRGB[2]  = frontRGB[2] + fDiff*sampleOp*sampleRGB[2];
            frontOp      = frontOp + fDiff*sampleOp;
            if (frontOp >= .95)
                break;
            
        }
         
        //store the composite in this pixel
        pixel[0] = frontRGB[0]/256.0;
        pixel[1] = frontRGB[1]/256.0;
        pixel[2] = frontRGB[2]/256.0;
        pixel[3] = 1.0;
    }//operator


    template <typename PosType,
              typename BBoxType, 
              typename IdxType>
    VTKM_EXEC
    vtkm::Float32 GetCellSample(const PosType      curPos, 
                                const BBoxType     bbox, 
                                const IdxType      idx) const
    {

        //LERP front face value
        vtkm::Int32 botFrontLeft  = idx[2]*dims[0]*dims[1]
                                   +idx[1]*dims[0]+idx[0];
        idx[0] += 1;
        vtkm::Int32 botFrontRight = idx[2]*dims[0]*dims[1]
                                   +idx[1]*dims[0]+idx[0];
        idx[1] += 1; 
        vtkm::Int32 topFrontRight = idx[2]*dims[0]*dims[1]
                                   +idx[1]*dims[0]+idx[0];
        idx[0] -= 1;
        vtkm::Int32 topFrontLeft  = idx[2]*dims[0]*dims[1]
                                   +idx[1]*dims[0]+idx[0];

        vtkm::Float32 botFrontVal = vtkm::Lerp(field.Get(botFrontLeft), field.Get(botFrontRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 topFrontVal = vtkm::Lerp(field.Get(botFrontLeft), field.Get(topFrontRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 frontVal    = vtkm::Lerp(botFrontVal, topFrontVal, 
                                              (curPos[1]-bbox[2])/(bbox[3] - bbox[2]));

    
        //LERP back face value    
        idx[1] -= 1;
        idx[2] += 1;
        vtkm::Int32 botBackLeft  = idx[2]*dims[0]*dims[1]
                                  +idx[1]*dims[0]+idx[0];
        idx[0] += 1;
        vtkm::Int32 botBackRight = idx[2]*dims[0]*dims[1]
                                  +idx[1]*dims[0]+idx[0];
        idx[1] += 1;
        vtkm::Int32 topBackRight = idx[2]*dims[0]*dims[1]
                                  +idx[1]*dims[0]+idx[0];
        idx[0] -= 1;
        vtkm::Int32 topBackLeft  = idx[2]*dims[0]*dims[1]
                                  +idx[1]*dims[0]+idx[0];
   
        vtkm::Float32 botBackVal = vtkm::Lerp(field.Get(botBackLeft), field.Get(botBackRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 topBackVal = vtkm::Lerp(field.Get(botBackLeft), field.Get(topBackRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 backVal    = vtkm::Lerp(botBackVal, topBackVal, 
                                              (curPos[1]-bbox[2])/(bbox[3] - bbox[2]));
    
        //LERP between the front and back faces
        return vtkm::Lerp(frontVal, backVal, (curPos[2]-bbox[4])/(bbox[5]-bbox[4]));
    }//GetCellSample


    template <typename BBoxType> 
    VTKM_EXEC
    void
    BoundingBoxForCell(const vtkm::Int32  cellId, 
                             BBoxType     bbox) const
    {
        bbox[0] = -1.0;
        bbox[1] = -1.0;
        bbox[2] = -1.0;
        bbox[3] = -1.0;
        bbox[4] = -1.0;
        bbox[5] = -1.0;
        if ((dims[0]-1)*(dims[1]-1)*(dims[2]-1) <= cellId || cellId < 0)
        {
            //cerr << "INVALID cellId!" << endl;
            return;
        }

        vtkm::Int32 idx[3];
        idx[0] = cellId%(dims[0]-1);
        idx[1] = (cellId/(dims[0]-1))%(dims[1]-1);
        idx[2] = cellId/((dims[0]-1)*(dims[1]-1));

        bbox[0] = coordPortals[0].Get(idx[0]);
        bbox[1] = coordPortals[0].Get(idx[0]+1);
        bbox[2] = coordPortals[1].Get(idx[1]);
        bbox[3] = coordPortals[1].Get(idx[1]+1);
        bbox[4] = coordPortals[2].Get(idx[2]);
        bbox[5] = coordPortals[2].Get(idx[2]+1);
    }//BoundingBoxForCell


    template <typename SearchPtType,
              typename IdxType>
    VTKM_EXEC
    void FindGridPoint(const SearchPtType &searchPt, 
                             IdxType      &ptIdx) const
    {
    
        
        //First check if the search point is even within the grid
        if ( (searchPt[0] < coordPortals[0].Get(0)) || (searchPt[0] > coordPortals[0].Get(dims[0]-1)) ||
             (searchPt[1] < coordPortals[1].Get(0)) || (searchPt[1] > coordPortals[1].Get(dims[1]-1)) || 
             (searchPt[2] < coordPortals[2].Get(0)) || (searchPt[2] > coordPortals[2].Get(dims[2]-1)) )
        {
            ptIdx[0] = -1;
            ptIdx[1] = -1;
            ptIdx[2] = -1;
            return;
        }
     
        BinarySearch(searchPt, ptIdx);
    }//FindGridPoint
                       

    template <typename PortalType,
              typename IntType,
              typename SearchPtType>
    VTKM_EXEC
    vtkm::Int32 SingleBinarySearch(const PortalType   &arr, 
                                   const IntType      &arr_len, 
                                   const SearchPtType &target) const
    {
        vtkm::Int32 s_min = 0;
        vtkm::Int32 s_max = arr_len - 1;
        vtkm::Int32 s_mid = s_max/2;

        while (s_min < s_max)
        {
            if (target == arr.Get(s_mid))
                return s_mid;

            if (s_max - s_min == 1)
                return s_min;

            if (target < arr.Get(s_mid))
            {
                s_max = s_mid;
                s_mid = (vtkm::Int32)((s_max - s_min + 1)/2) + s_min;
            }

            if (target > arr.Get(s_mid))
            { 
                s_min = s_mid;
                s_mid = (vtkm::Int32)((s_max - s_min + 1)/2) + s_min;
            }
        }

        return s_mid - 1;
    }//SingleBinarySearch


    /**
    * Perform a binary search through the data set. We want to find 
    * where in our rectilinear grid a given point resides. 
    * I'm finding the lower left corner of the associated cell. 
    *
    **/
    template <typename SearchPtType,
              typename IdxType>
    VTKM_EXEC
    void BinarySearch(const SearchPtType &searchPt, 
                            IdxType      &ptIdx) const
    {

        ptIdx[0] = SingleBinarySearch(coordPortals[0], dims[0], searchPt[0]);
        ptIdx[1] = SingleBinarySearch(coordPortals[1], dims[1], searchPt[1]);
        ptIdx[2] = SingleBinarySearch(coordPortals[2], dims[2], searchPt[2]);
    }//BinarySearch

};//RayCaster
}//worklet
}//vtkm


int main()
{
    
    vtkm::io::reader::VTKDataSetReader reader("astro512.vtk");
    vtkm::cont::DataSet inData = reader.ReadDataSet();
    
    //Set up a vtkm canvas and get the pixel buffer
    vtkm::rendering::CanvasRayTracer canvas(W, H);
    vtkm::Vec<vtkm::Float32, 4> * buffer = canvas.GetColorBuffer().GetStorage().GetArray();

    vtkm::cont::DynamicArrayHandleCoordinateSystem dcoords = inData.GetCoordinateSystem().GetData();
    vtkm::cont::ArrayHandleCartesianProduct<vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                            vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                            vtkm::cont::ArrayHandle<vtkm::FloatDefault>>
      coordinates;

    if (dcoords.IsSameType(vtkm::cont::ArrayHandleUniformPointCoordinates()))
    {
        cerr << "FIRST" << endl;
        cerr << "NOT IMPLEMENTED" << endl;
        return 1;
    }
    else if (dcoords.IsSameType(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3>>()))
    {
        cerr << "SECOND" << endl;
        cerr << "NOT IMPLEMENTED" << endl;
        return 1;
    }
    else if (dcoords.IsSameType(vtkm::cont::ArrayHandleCartesianProduct<
                                vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                vtkm::cont::ArrayHandle<vtkm::FloatDefault>>()))
    {
        coordinates = dcoords.Cast<
        vtkm::cont::ArrayHandleCartesianProduct<vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                                vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
                                                vtkm::cont::ArrayHandle<vtkm::FloatDefault>>>();
    }
    else
    {
        cerr << "NONE" << endl;
        return 1;
    }

    //TransferFunction tf = SetupTransferFunction();
    vtkm::exec::TransferFunction tf;

    //create screen
    vtkm::exec::Screen screen;
    screen.SetDimensions(W, H);
    screen.Initialize();

    //create camera
    vtkm::exec::Camera camera;
    camera.InitCamera();
 
    //get field
    vtkm::cont::ArrayHandle<vtkm::Float32> field;
    field = inData.GetField(0).GetData().Cast<vtkm::cont::ArrayHandle<vtkm::Float32>>();

    //Invoke the worklet
    vtkm::worklet::RayCaster worklet(coordinates, field);
    typedef vtkm::worklet::DispatcherMapField<vtkm::worklet::RayCaster> dispatcher;  
    
    //get timing info
    struct timespec start, end;

    if (TIMING)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }

    dispatcher(worklet).Invoke(canvas.GetColorBuffer(), camera, screen, tf);

    if (TIMING)
    {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed;
        elapsed  = (end.tv_sec - start.tv_sec);
        elapsed += (end.tv_nsec - start.tv_nsec)/1000000000.0;
        cerr << "elapsed time: " << elapsed << endl;
    }

    //write image buffer
    canvas.SaveAs("out.ppm");

}
