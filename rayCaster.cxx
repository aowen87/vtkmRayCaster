/***
* @author: Alister Maguire
*
* This is a conversion of my ray casting algorithm 
* to rely on VTK-M. NOTE: It is currently set up
* for one specific input (the camera position and
* transfer function are specific to this data set). 
*
***/
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <vector>

#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkDataSetReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkDataSetWriter.h>
#include <vtkPNGWriter.h>

#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/exec/ExecutionObjectBase.h>
#include <vtkm/VectorAnalysis.h>
#include <vtkm/Math.h>

#include <vtkm/rendering/CanvasRayTracer.h>


using std::cerr;
using std::cout;
using std::endl;

#define TIMING false
#define NUM_SAMPLES 500
#define BASE_SAMP_RATE 200
#define PI 3.14159
#define H 500
#define W 500



vtkImageData *NewImage(int width, int height)
{
    vtkImageData *img = vtkImageData::New();
    img->SetDimensions(width, height, 1);
    img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    return img;
}


void WriteImage(vtkImageData *img, const char *filename)
{
   std::string full_filename = filename;
   full_filename += ".png";
   vtkPNGWriter *writer = vtkPNGWriter::New();
   writer->SetInputData(img);
   writer->SetFileName(full_filename.c_str());
   writer->Write();
   writer->Delete();
}


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

  public:

    typedef void ControlSignature(FieldOut<>     pixels,
                                  WholeArrayIn<> X,
                                  WholeArrayIn<> Y,
                                  WholeArrayIn<> Z,
                                  WholeArrayIn<> F,
                                  WholeArrayIn<> dims,
                                  ExecObject     Camera,
                                  ExecObject     Screen,
                                  ExecObject     TransferFunction);

    typedef void ExecutionSignature(_1,
                                    _2,
                                    _3,
                                    _4,
                                    _5,
                                    _6,
                                    _7,
                                    _8,
                                    _9,
                                    InputIndex);
    typedef _1 InputDomain;

    template<typename CameraType,
             typename ScreenType,
             typename PixelType,
             typename FloatVecType,
             typename IntVecType,
             typename TransferFunctionType>
    VTKM_EXEC
    void operator() (PixelType                  &pixel,
                     const FloatVecType         &X,
                     const FloatVecType         &Y,
                     const FloatVecType         &Z,
                     const FloatVecType         &F,
                     const IntVecType           &dims,
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

            FindGridPoint(X, Y, Z, F, dims, curPos, ptIdx);
            vtkm::Int32 cellId = ptIdx[2]*(dims.Get(0)-1)*(dims.Get(1)-1)
                                +ptIdx[1]*(dims.Get(0)-1)+ptIdx[0];


            vtkm::Float32 bbox[6];
            BoundingBoxForCell(X, Y, Z, dims, cellId, bbox);
           
             
            if (bbox[0] == -1.0)
            {
               samples[i] = 0.0;
               curPos    += stepSize*ray;
               ++i; 
               continue;
            }

            
            samples[i] = GetCellSample(X, Y, Z, F, dims, curPos, bbox, ptIdx);
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
        //pixel = vtkm::make_Vec(frontRGB[0], frontRGB[1], frontRGB[2]);
        pixel[0] = frontRGB[0]/256.0;
        pixel[1] = frontRGB[1]/256.0;
        pixel[2] = frontRGB[2]/256.0;
        pixel[3] = 1.0;
      
    }//operator


    template <typename FloatVecType,
              typename IntVecType,
              typename PosType,
              typename BBoxType, 
              typename IdxType>
    VTKM_EXEC
    vtkm::Float32 GetCellSample(const FloatVecType X, 
                                const FloatVecType Y, 
                                const FloatVecType Z, 
                                const FloatVecType F, 
                                const IntVecType   dims, 
                                const PosType      curPos, 
                                const BBoxType     bbox, 
                                const IdxType      idx) const
    {

        //LERP front face value
        vtkm::Int32 botFrontLeft  = idx[2]*dims.Get(0)*dims.Get(1)
                                   +idx[1]*dims.Get(0)+idx[0];
        idx[0] += 1;
        vtkm::Int32 botFrontRight = idx[2]*dims.Get(0)*dims.Get(1)
                                   +idx[1]*dims.Get(0)+idx[0];
        idx[1] += 1; 
        vtkm::Int32 topFrontRight = idx[2]*dims.Get(0)*dims.Get(1)
                                   +idx[1]*dims.Get(0)+idx[0];
        idx[0] -= 1;
        vtkm::Int32 topFrontLeft  = idx[2]*dims.Get(0)*dims.Get(1)
                                   +idx[1]*dims.Get(0)+idx[0];

        vtkm::Float32 botFrontVal = vtkm::Lerp(F.Get(botFrontLeft), F.Get(botFrontRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 topFrontVal = vtkm::Lerp(F.Get(botFrontLeft), F.Get(topFrontRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 frontVal    = vtkm::Lerp(botFrontVal, topFrontVal, 
                                              (curPos[1]-bbox[2])/(bbox[3] - bbox[2]));

    
        //LERP back face value    
        idx[1] -= 1;
        idx[2] += 1;
        vtkm::Int32 botBackLeft  = idx[2]*dims.Get(0)*dims.Get(1)
                                  +idx[1]*dims.Get(0)+idx[0];
        idx[0] += 1;
        vtkm::Int32 botBackRight = idx[2]*dims.Get(0)*dims.Get(1)
                                  +idx[1]*dims.Get(0)+idx[0];
        idx[1] += 1;
        vtkm::Int32 topBackRight = idx[2]*dims.Get(0)*dims.Get(1)
                                  +idx[1]*dims.Get(0)+idx[0];
        idx[0] -= 1;
        vtkm::Int32 topBackLeft  = idx[2]*dims.Get(0)*dims.Get(1)
                                  +idx[1]*dims.Get(0)+idx[0];
   
        vtkm::Float32 botBackVal = vtkm::Lerp(F.Get(botBackLeft), F.Get(botBackRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 topBackVal = vtkm::Lerp(F.Get(botBackLeft), F.Get(topBackRight), 
                                              (curPos[0]-bbox[0])/(bbox[1] - bbox[0]));
        vtkm::Float32 backVal    = vtkm::Lerp(botBackVal, topBackVal, 
                                              (curPos[1]-bbox[2])/(bbox[3] - bbox[2]));
    
        //LERP between the front and back faces
        return vtkm::Lerp(frontVal, backVal, (curPos[2]-bbox[4])/(bbox[5]-bbox[4]));
    }//GetCellSample


    template <typename FloatVecType,
              typename IntVecType,
              typename BBoxType> 
    VTKM_EXEC
    void
    BoundingBoxForCell(const FloatVecType X,
                        const FloatVecType Y, 
                        const FloatVecType Z, 
                        const IntVecType   dims,
                        const vtkm::Int32  cellId, 
                              BBoxType     bbox) const
    {
        bbox[0] = -1.0;
        bbox[1] = -1.0;
        bbox[2] = -1.0;
        bbox[3] = -1.0;
        bbox[4] = -1.0;
        bbox[5] = -1.0;
        if ((dims.Get(0)-1)*(dims.Get(1)-1)*(dims.Get(2)-1) <= cellId || cellId < 0)
        {
            //cerr << "INVALID cellId!" << endl;
            return;
        }

        vtkm::Int32 idx[3];
        idx[0] = cellId%(dims.Get(0)-1);
        idx[1] = (cellId/(dims.Get(0)-1))%(dims.Get(1)-1);
        idx[2] = cellId/((dims.Get(0)-1)*(dims.Get(1)-1));

        bbox[0] = X.Get(idx[0]);
        bbox[1] = X.Get(idx[0]+1);
        bbox[2] = Y.Get(idx[1]);
        bbox[3] = Y.Get(idx[1]+1);
        bbox[4] = Z.Get(idx[2]);
        bbox[5] = Z.Get(idx[2]+1);
    }//BoundingBoxForCell


    template <typename FloatVecType,
              typename IntVecType,
              typename SearchPtType,
              typename IdxType>
    VTKM_EXEC
    void FindGridPoint(const FloatVecType &X, 
                        const FloatVecType &Y, 
                        const FloatVecType &Z, 
                        const FloatVecType &F, 
                        const IntVecType   &dims,
                        const SearchPtType &searchPt, 
                              IdxType      &ptIdx) const
    {
    
        
        //First check if the search point is even within the grid
        if ( (searchPt[0] < X.Get(0)) || (searchPt[0] > X.Get(dims.Get(0)-1)) ||
             (searchPt[1] < Y.Get(0)) || (searchPt[1] > Y.Get(dims.Get(1)-1)) || 
             (searchPt[2] < Z.Get(0)) || (searchPt[2] > Z.Get(dims.Get(2)-1)) )
        {
            ptIdx[0] = -1;
            ptIdx[1] = -1;
            ptIdx[2] = -1;
            return;
        }
     
        BinarySearch(X, Y, Z, dims, searchPt, ptIdx);
    }//FindGridPoint
                       

    template <typename FloatVecType,
              typename IntType,
              typename SearchPtType>
    VTKM_EXEC
    vtkm::Int32 SingleBinarySearch(const FloatVecType &arr, 
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
    template <typename FloatVecType,
              typename IntVecType,
              typename SearchPtType,
              typename IdxType>
    VTKM_EXEC
    void BinarySearch(const FloatVecType &X, 
                       const FloatVecType &Y, 
                       const FloatVecType &Z, 
                       const IntVecType   &dims, 
                       const SearchPtType &searchPt, 
                             IdxType      &ptIdx) const
    {

        ptIdx[0] = SingleBinarySearch(X, dims.Get(0), searchPt[0]);
        ptIdx[1] = SingleBinarySearch(Y, dims.Get(1), searchPt[1]);
        ptIdx[2] = SingleBinarySearch(Z, dims.Get(2), searchPt[2]);
    }//BinarySearch

};//RayCaster
}//worklet
}//vtkm


int main()
{

    
    //vtkm::io::reader::VTKDataSetReader reader("astro512.vtk");
    //vtkm::cont::DataSet inData = reader.ReadDataSet();
    
    //vtkm::rendering::CanvasGL canvas;//(1000, 1000);

    //Set up a vtkm canvas and get the pixel buffer
    vtkm::rendering::CanvasRayTracer canvas(W, H);
    vtkm::Vec<vtkm::Float32, 4> * buffer = canvas.GetColorBuffer().GetStorage().GetArray();
  
    //Set up transfer function
    //TransferFunction tf = SetupTransferFunction();
    vtkm::exec::TransferFunction tf;
    //read in data
    vtkDataSetReader *rdr = 
     vtkDataSetReader::New();
    rdr->SetFileName("astro512.vtk");
    rdr->Update();


    //Get the rectilinear grid from the file
    vtkSmartPointer<vtkRectilinearGrid> rgrid = (vtkRectilinearGrid *)rdr->GetOutput();
    vtkm::Int32 dims[3];
    rgrid->GetDimensions(dims); 

    //Get data from the rgrid
    static vtkm::Float32 *X = (float *) rgrid->GetXCoordinates()->GetVoidPointer(0);
    static vtkm::Float32 *Y = (float *) rgrid->GetYCoordinates()->GetVoidPointer(0);
    static vtkm::Float32 *Z = (float *) rgrid->GetZCoordinates()->GetVoidPointer(0);
    static vtkm::Float32 *F = (float *) rgrid->GetPointData()->GetScalars()->GetVoidPointer(0);

    //Set up the image
    vtkm::Int32 width  = W;
    vtkm::Int32 height = H;
    vtkImageData  *image  = NewImage(width, height);
    //unsigned char *buffer = 
    //  (unsigned char *) image->GetScalarPointer(0,0,0);
    
    vtkm::Int32 npixels = width*height;
    
    //vtkm code starts here

    //create screen
    vtkm::exec::Screen screen;
    screen.SetDimensions(width, height);
    screen.Initialize();

    //create camera
    vtkm::exec::Camera camera;
    camera.InitCamera();

    //create array handles 
    vtkm::cont::ArrayHandle<vtkm::Float32> XHandle =
        vtkm::cont::make_ArrayHandle(X, dims[0]);
    vtkm::cont::ArrayHandle<vtkm::Float32> YHandle =
        vtkm::cont::make_ArrayHandle(Y, dims[1]);
    vtkm::cont::ArrayHandle<vtkm::Float32> ZHandle =
        vtkm::cont::make_ArrayHandle(Z, dims[2]);
    vtkm::cont::ArrayHandle<vtkm::Float32> FHandle =
        vtkm::cont::make_ArrayHandle(F, dims[0]*dims[1]*dims[2]);

    /*
    std::vector<vtkm::Vec<unsigned char, 3>> PBuff(width*height);
    vtkm::cont::ArrayHandle<vtkm::Vec<unsigned char, 3>> PHandle =
        vtkm::cont::make_ArrayHandle(PBuff);
    */
    vtkm::cont::ArrayHandle<vtkm::Int32> DHandle =
        vtkm::cont::make_ArrayHandle(dims, 3);

    //Invoke the worklet
    vtkm::worklet::RayCaster worklet;
    typedef vtkm::worklet::DispatcherMapField<vtkm::worklet::RayCaster> dispatcher;  
    //dispatcher(worklet).Invoke(PHandle, XHandle, YHandle, ZHandle, FHandle, DHandle, 
    //                           camera, screen, tf);
    dispatcher(worklet).Invoke(canvas.GetColorBuffer(), XHandle, YHandle, ZHandle, FHandle, DHandle, 
                               camera, screen, tf);

    //Copy the worklet buffer voer to the image buffer
    /*
    for (int i = 0; i < width*height; ++i)
    {
        buffer[3*i]   = PHandle.GetPortalConstControl().Get(i)[0];
        buffer[3*i+1] = PHandle.GetPortalConstControl().Get(i)[1];
        buffer[3*i+2] = PHandle.GetPortalConstControl().Get(i)[2];
    }

    WriteImage(image, "myOut");
    image->Delete();
    */ 
    canvas.SaveAs("out.ppm");
}
