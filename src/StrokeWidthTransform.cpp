#include "StrokeWidthTransform.h"
#include "fstream"

#define DEBUG 0
SWTransform::SWTransform(cv::Mat & srcColor, int direction, double thresholdRatio)
{
        maxStrokeWidth = std::max(srcColor.rows, srcColor.cols);
        initialSW = maxStrokeWidth * 2;
        this->direction = direction;
        this->thresholdRatio = thresholdRatio;
        
        theta = cv::Mat(srcColor.size(), CV_32FC1, cv::Scalar(0));
        strokeW = cv::Mat(srcColor.size(), CV_32FC1, cv::Scalar(initialSW));
        origColor = srcColor.clone();
        src = srcColor.clone();
        PreProcessor::toGrayScale(src);
        startPoints.clear();
}
/*!
 * Finds the perpendicular pixel according to the gradient direction(angle) at the current pixel
 * curRow, curCol - set pixels in edge image
 * nextRow, nextCol - the pixel in perpendicular direction
 * step - Denotes how far from (curRow, curCol) the new pixel should be
 * direction - either +1 or -1
 */
void SWTransform::nextCoOrdinate(float curRow, float curCol, float & nextRow, float & nextCol, double angle, int step, int direction)
{
        /*
         *  Refer to 
         */
        nextRow = round(curRow + step * sin(angle) * direction);
        nextCol = round(curCol + step * cos(angle) * direction);
}

/*!
 * Just a debug function
 * Prints the 3x3 values of src centered at (row, col)
 */
template<class T, class V> 
void print33(cv::Mat & src, int row, int col)
{
        if(!(row - 1 > 0 && row + 1 < src.rows && col - 1 > 0 && col + 1 < src.cols))
        {
                //No can't print the grid sire. 
                //You just gave me wrong co-ordinates. Shame on you.
                return;
        }
        std::cout<<"At "<<"("<<row<<", "<<col<<")\n";
        std::cout<<(V)src.at<T>(row - 1, col - 1)<<" ";
        std::cout<<(V)src.at<T>(row - 1, col)<<" ";
        std::cout<<(V)src.at<T>(row - 1, col + 1)<<"\n";
        
        std::cout<<(V)src.at<T>(row, col - 1)<<" ";
        std::cout<<(V)src.at<T>(row, col)<<" ";
        std::cout<<(V)src.at<T>(row, col + 1)<<"\n";

        std::cout<<(V)src.at<T>(row + 1, col - 1)<<" ";
        std::cout<<(V)src.at<T>(row + 1, col)<<" ";
        std::cout<<(V)src.at<T>(row + 1, col + 1)<<"\n";
}

/*!
 * Outer function to calculate SWT
 * calls doStrokeWidth() after finding the set pixels in edge img
 */

void SWTransform::calculateStrokeWidth(cv::Mat & edgeImg)
{
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - Entering SWT\n";
                std::cout<<"DEBUG: SWT - Writing refined edge image out as ImgEdge.jpg\n";
                GenericHelpers::Display("ImgEdge", edgeImg, 1);
                std::cout<<"DEBUG: SWT - Finding set pixels in CANNY\n";
        }
        
        //Sobel gradients - used to find the direction of gradient at a pixel
        cv::Mat grad_x, grad_y;
        GenericHelpers::sobel(src, grad_x, grad_y);
        std::set<float> strokewidths;
        
        for(int i = 0; i < src.rows; ++i)
        {
                for(int j = 0; j < src.cols; ++j)
                {
                        if(edgeImg.at<uchar>(i, j) != 0)
                        {
                                startPoints.push_back(cv::Point(j, i));
                                float angle = atan2(grad_y.at<float>(i, j), grad_x.at<float>(i, j));
                                theta.at<float>(cv::Point(j, i)) = angle;
                        }

                }
        }
        doStrokeWidth(edgeImg, false);
        doStrokeWidth(edgeImg, true);
       
        cv::Mat writ(strokeW.size(), CV_8UC1, cv::Scalar(0));
        for(int i = 0; i < strokeW.rows; ++i)
        {
                for(int j = 0; j < strokeW.cols; ++j)
                {
                        writ.at<uchar>(i, j) = (strokeW.at<float>(i, j) > 0.0 ? 255 : 0);
                        strokewidths.insert(strokeW.at<float>(i, j));
                }
        }

        /*std::ofstream fi;
        fi.open("output.txt", std::ios_base::app);
        for(std::set<float>::iterator it = strokewidths.begin(); it != strokewidths.end(); ++it)
        {
                if(*it > 0.0)
                        fi<<*it<<"\n";
        }
        fi<<"----------------\n";
        fi.close();*/
        std::stringstream ss;
        ss<<"SWTV"<<direction<<".jpg";
        cv::imwrite(ss.str(), writ);
}

/*!
 * Actual function where StrokeWidthTransform(SWT) magic happens.
 * src - source image - 8bit 1channel
 * startPoints - co-ordinates set pixels in edgeImage
 * theta - contains gradient direction at each pixel
 * strokeW - resultant Mat into which stroke widths have to be put into
 * edgeImg - an AND image of Canny and Gabor Images
 * direction - +90 or -90
 */
void SWTransform::doStrokeWidth(cv::Mat & edgeImg, bool isMedian)
{
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - Actually performing SWT\n";

                //Maximum length a stroke can take.
                //TODO: How do you determine its value for different input images?
                std::cout<<"DEBUG: SWT - Max Stroke Width = "<<maxStrokeWidth<<std::endl;
        }

        std::vector<cv::Point> points;

        for(int i = 0; i < startPoints.size(); ++i)
        {
                float row = startPoints[i].y;
                float column = startPoints[i].x;
                float iX = column, iY = row;

                //We are starting a new ray, so discard the old points and push the current pixel from edgeImage
                points.clear();
                points.push_back(cv::Point(iX, iY));
                std::vector<double> swtValues;
                swtValues.push_back(strokeW.at<float>(row, column));

                float angle = theta.at<float>(startPoints[i]);
                float origAngle = angle; //Store the original gradient angle for later comparison
                float fRow = -1; //final Row
                float fCol = -1; // final Column
                int step = 1;
                while(step < maxStrokeWidth)
                {
                        float nextRow, nextCol;
                        nextCoOrdinate(iY, iX, nextRow, nextCol, angle, step, direction);

                        //if(iX == 516 && iY == 80) std::cout<<"DEBUG: NEXT - "<<nextRow<<" "<<nextCol<<std::endl;
                        if(nextRow < 0 || nextRow >= edgeImg.rows || nextCol < 0 || nextCol >= edgeImg.cols) 
                        {
                                break;
                        }
                        step++;
                        if(row == nextRow && column == nextCol) 
                        {
                                //We are stuck at the same pixel -_-
                                //Increase step value and continue
                                continue;
                        }
                        bool hitEdgePixel = false;
                        hitEdgePixel = (hitEdgePixel || (edgeImg.at<uchar>(nextRow, nextCol) != 0));

                        /*if(row != nextRow && column != nextCol)
                        {
                                uchar pix1, pix2;
                                if(nextCol == column - 1)
                                {
                                        pix1 = edgeImg.at<uchar>(row, column - 1);
                                }
                                else
                                {
                                        pix1 = edgeImg.at<uchar>(row, column + 1);
                                }
                                if(nextRow == row - 1)
                                {
                                        pix2 = edgeImg.at<uchar>(row - 1, column);
                                }
                                else
                                {
                                        pix2 = edgeImg.at<uchar>(row + 1, column);
                                }
                                hitEdgePixel = (hitEdgePixel || (pix1 != 0 && pix2 != 0));
                        }*/
                        row = nextRow;
                        column = nextCol;
                        points.push_back(cv::Point(column, row));
                        swtValues.push_back(strokeW.at<float>(row, column));
                        float curAngle = theta.at<float>(cv::Point(column, row));

                        //I have travelled along the direction of the ray and hit some other edge pixel
                                
                        if(hitEdgePixel) 
                        {
                                //Is the current pixel angle roughly perpendicular to the current pixel angle?
                                if((std::abs(std::abs(curAngle - origAngle) - 3.14)) < 3.14/2)
                                {
                                        fRow = row;
                                        fCol = column;
                                }
                                break;
                        }
                }
                //So this is a valid ray. Update the stroke width of pixels
                if(fRow != -1 && fCol != -1)
                {
                        /*
                         * We update the row and column values to take into account the 
                         * end pixel value too, while computing the distance
                         */
                        //We are in the same row, so its enough if we increment the column
                        if(fRow == startPoints[i].y)
                        {
                                if(fCol > startPoints[i].x)
                                {
                                        fCol += 1.0;
                                }
                                else
                                {
                                        fCol -= 1.0;
                                }
                        }
                        else if(fCol == startPoints[i].x)
                        {
                                //In the same column, so increment the row
                                if(fRow > startPoints[i].y)
                                {
                                        fRow += 1.0;
                                }
                                else
                                {
                                        fRow -= 1.0;
                                }
                        }
                        else
                        { 
                                //Not parallel or perpendicular ray, so increment both row and column
                                if(fRow > startPoints[i].y && fCol > startPoints[i].x)
                                {
                                        fRow += 1.0;
                                        fCol += 1.0;
                                }
                                else if(fCol < startPoints[i].x && fRow < startPoints[i].y)
                                {
                                        fRow -= 1.0;
                                        fCol -= 1.0;
                                }
                        }
                        //Current stroke width(represented by dis)
                        float dis = (float)sqrt((float)(fRow - startPoints[i].y) * (fRow - startPoints[i].y) + (float)(fCol - startPoints[i].x) * (fCol - startPoints[i].x));

                        if(isMedian)
                        {
                                std::nth_element(swtValues.begin(), swtValues.begin() + swtValues.size() / 2, swtValues.end());
                                dis = swtValues[swtValues.size() / 2];
                        }
                        for(int k = 0; k < points.size(); ++k)
                        {
                                strokeW.at<float>(points[k]) = std::min(strokeW.at<float>(points[k]), dis);
                                //if(points[k].y == 83 && points[k].x == 527) std::cout<<"DEBUG: "<<iX<<" "<<iY<<std::endl;
                        }
                        rays.push_back(points);
                }
        }
       
        //Every pixel in strokeW image that does not have a stroke associated with it, will have value 0
        for(int i = 0; i < strokeW.rows; ++i)
        {
                for(int j = 0; j < strokeW.cols; ++j)
                {
                        if(strokeW.at<float>(i, j) == initialSW)
                        {
                                strokeW.at<float>(i, j) = 0;
                        }
                }
        }
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - End of doStrokeWidth\n";
        }
}
void SWTransform::findStrokeConnectedComponents()
{
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - Finding connected components in Stroke Width Transform Image\n";
        }
        assert(strokeW.depth() == CV_32F && strokeW.channels() == 1);

        labelledImg = cv::Mat(strokeW.size(), CV_32FC1, cv::Scalar(-1.0));
        int arr1[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int arr2[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        int label = 0;
        
        for(int i = 0; i < strokeW.rows; ++i)
        {
                for(int j = 0; j < strokeW.cols; ++j)
                {
                        bool connected = false;
                        if(strokeW.at<float>(i, j) > 0.0 && labelledImg.at<float>(i, j) == -1.0)
                        {
                                std::stack< std::pair<int, int> > st;
                                st.push( std::make_pair(i, j) );
                                
                                std::vector<cv::Point> points;
                                while(!st.empty())
                                {
                                        int cRow = st.top().first;
                                        int cCol = st.top().second;
                                        points.push_back(cv::Point(cCol, cRow));
                                        st.pop();
                              
                                        for(int k = 0; k < 8; ++k)
                                        {
                                                int dRow = cRow + arr1[k];
                                                int dCol = cCol + arr2[k];

                                                if(dRow < 0 || dRow >= strokeW.rows || dCol < 0 || dCol >= strokeW.cols)
                                                {
                                                        continue;
                                                }
                                        
                                                if(strokeW.at<float>(dRow, dCol) == 0)
                                                {
                                                        labelledImg.at<float>(dRow, dCol) = -2.0;
                                                        continue;
                                                }
                                                if(labelledImg.at<float>(dRow, dCol) == -1.0)
                                                {
                                                        float a = strokeW.at<float>(dRow, dCol);
                                                        float b = strokeW.at<float>(cRow, cCol);

                                                        if(std::max(a, b) / std::min(a, b) <= thresholdRatio)
                                                        {
                                                                labelledImg.at<float>(dRow, dCol) = label;
                                                                st.push( std::make_pair(dRow, dCol) );
                                                                connected = true;
                                                        }
                                                }
                                        }
                                }
                                if(connected)
                                {
                                        connectedComponent cc;
                                        cc.strokes = points;
                                        ccs.push_back(cc);
                                        label++;
                                }
                                else
                                {
                                        labelledImg.at<float>(i, j) = -2.0;
                                }
                        }
                }
        }
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - Total components - "<<label<<std::endl;
        }

        findBoundingBoxes();
#if 0
        int rowSt = 31, rowEd = 33;
        int colSt = 221, colEd = 223;
        for(int i = rowSt; i <= rowEd; ++i)
        {
                for(int j = colSt; j <= colEd; ++j)
                {
                        std::cout<<(int)strokeW.at<float>(i, j)<<" ";
                }
                std::cout<<std::endl;
        }
#endif
}

void SWTransform::varianceFilter()
{
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - In Variance Filter\n";
                std::cout<<"DEBUG: SWT - CCS Size - "<<ccs.size()<<"\n";
        }
        std::vector<connectedComponent> filteredCC;
        for(int i = 0; i < ccs.size(); ++i)
        {
                connectedComponent curCC = ccs[i];
                float swMin = 5005, swMax = 0;
                double sumWidth = 0.0;
                for(int j = 0; j < curCC.strokes.size(); ++j)
                {
                        swMin = std::min(swMin, strokeW.at<float>(curCC.strokes[j]));
                        swMax = std::max(swMax, strokeW.at<float>(curCC.strokes[j]));
                        sumWidth += strokeW.at<float>(curCC.strokes[j]);
                }
                double mean = (sumWidth / (curCC.strokes.size()));

                double variance = 0;
                for(int j = 0; j < curCC.strokes.size(); ++j)
                {
                        double val = strokeW.at<float>(curCC.strokes[j]) - mean;
                        variance += (val * val);
                }
                variance /= curCC.strokes.size();

                double sd = sqrt(variance);
                if(variance / mean < 1.5)
                {
                        filteredCC.push_back(ccs[i]);
                }
                else
                {
                        for(int j = 0; j < curCC.strokes.size(); ++j)
                        {
                                labelledImg.at<float>(curCC.strokes[j]) = -1.0;
                        }
                }
        }
        ccs = filteredCC; 
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - After filtering CCS Size - "<<ccs.size()<<"\n";
        }
        findBoundingBoxes();
}

void SWTransform::findBoundingBoxes()
{
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - Finding bounding boxes for connected components\n";
                std::cout<<"DEBUG: SWT - Total connected components - "<<ccs.size()<<"\n";
        }
        roi.clear();
        for(int i = 0; i < ccs.size(); ++i)
        {
                cv::Rect cur;
                int x1 = 50000, y1 = 50000, x2 = 0, y2 = 0;
                for(int j = 0; j < ccs[i].strokes.size(); ++j)
                {
                        x1 = std::min(x1, ccs[i].strokes[j].x);
                        x2 = std::max(x2, ccs[i].strokes[j].x);
                        y1 = std::min(y1, ccs[i].strokes[j].y);
                        y2 = std::max(y2, ccs[i].strokes[j].y);
                }
                cur.x = x1;
                cur.y = y1;
                cur.width = x2 - x1 + 1;
                cur.height = y2 - y1 + 1;
                roi.push_back(cur);
        }
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - End of finding bounding boxes. Total ROI - "<<roi.size()<<"\n"; 
        }
}

void SWTransform::boundingBoxHeuristics()
{
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - Performing bounding box heuristics\n";
                std::cout<<"DEBUG: SWT - Total Regions Of Interest - "<<roi.size()<<"\n";
        }
        std::vector<cv::Rect> cleanRoi;
        for(int i = 0; i < roi.size(); ++i)
        {
                bool isCorrect = true;
                float maxSW = 0.0;
                int thisComponentPixelCount = 0;
                for(int j = roi[i].y; j < roi[i].y + roi[i].height; ++j)
                {
                       for(int k = roi[i].x; k < roi[i].x + roi[i].width; ++k)
                       {
                               if(labelledImg.at<float>(j, k) == i)
                               {
                                       maxSW = std::max(maxSW, strokeW.at<float>(j, k));
                                       thisComponentPixelCount++;
                               }
                       }
                }
                double aspectRatio = sqrt((double)roi[i].width * (double)roi[i].width / ((double)roi[i].height * (double)roi[i].height));
                aspectRatio /= maxSW;

                isCorrect = isCorrect && (aspectRatio < 10.0);
                isCorrect = isCorrect && (thisComponentPixelCount / maxSW > 5);
                isCorrect = isCorrect && (roi[i].width < 2.5 * roi[i].height);

                isCorrect = isCorrect && (roi[i].width * roi[i].height > 25);
                
                /* If you are uncommenting this portion, I can understand how desperate you are.
                 * Do not worry, you are not alone.
                 *
                 * int maxDimension = std::max(labelledImg.rows, labelledImg.cols);
                 * if(maxDimension >= 500)
                 * {
                 *       if(roi[i].width < 10 || roi[i].height < 10) isCorrect = false;
                 * }
                 */
                std::set<int> components;
                for(int j = roi[i].y; j < roi[i].y + roi[i].height; ++j)
                {
                       for(int k = roi[i].x; k < roi[i].x + roi[i].width; ++k)
                       {
                               if(labelledImg.at<float>(j, k) >= 0.0)
                               {
                                       components.insert(static_cast<int>(labelledImg.at<float>(j, k)));
                               }
                       }
                }
                if(components.size() > 6)
                {
                        isCorrect = false;
                        discardedRoi.push_back(roi[i]);
                }
                if(isCorrect)
                {
                         cleanRoi.push_back(roi[i]);
                }
        }
        roi = cleanRoi;
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - End of bounding box heuristics. Total ROI - "<<roi.size()<<"\n";
        }
}

void SWTransform::findPredominantComponent()
{
        if(DEBUG)
        {
                std::cout<<"DEBUG: SWT - Getting back some Rect's that were falsely discarded due to presence \
of other components(contribution threshold = 0.9)\n";
        }
        for(int i = 0; i < discardedRoi.size(); ++i)
        {
                cv::Rect curRoi = discardedRoi[i];

                std::map<float, int> componentsPresent;
                int totNonZero = 0;
                for(int j = curRoi.y; j < curRoi.y + curRoi.height; ++j)
                {
                        for(int k = curRoi.x; k < curRoi.x + curRoi.width; ++k)
                        {
                                float compLabel = labelledImg.at<float>(j, k);
                                if(compLabel >= 0.0)
                                {
                                        totNonZero++;
                                        if(componentsPresent.find(compLabel) != componentsPresent.end())
                                        {
                                                componentsPresent[compLabel]++;
                                        }
                                        else
                                        {
                                                componentsPresent[compLabel] = 1;
                                        }
                                }
                        }
                }
                
                std::vector<int> count;
                for(std::map<float, int>::iterator it = componentsPresent.begin(); it != componentsPresent.end(); ++it)
                {
                        count.push_back(it->second);
                }
                std::sort(count.rbegin(), count.rend());
                double thresh = 0.9;
                if(totNonZero != 0)
                {
                        double ratio = (double)count[0] / (double)totNonZero;
                        if(ratio >= thresh)
                        {
                                roi.push_back(curRoi);
                        }
                }
        }
}
