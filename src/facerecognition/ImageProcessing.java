/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facerecognition;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author NguyenVanDung
 */
public class ImageProcessing {

    private static Matrix meanMatrix;
    private static Matrix AA;
    private static Matrix symMatrix;
    private static Matrix eigenFace;
    //Luu giu trong so cua cac anh trong khong gian vector rieng
    //Ta de bo trong so theo chieu ngang
    private static double WW[][];
    //kich co anh la 92 x 112
    private static int wi = 92;
    private static int he = 112;
    //So luong anh trong tap luyen
    private static int N;

    /**
     * @param args the command line arguments
     */
    //   public static void main(String[] args) {
//        int N = 5;
//        double[][] data = new double[N][N];
//        double[][] beta = new double[N][N];
//        for (int i = 0; i < N; i++) {
//            for (int j = 0; j < N; j++) {
//                data[i][j] = i + j;
//                beta[i][j] = i * i + j * j;
//            }
//        }
//        Matrix matrix = new Matrix(data);
//       // matrix.norm1();
//        matrix.norm2();
//        matrix.normF();
//        matrix.normInf();
//        matrix.plusEquals(new Matrix(beta));
//        matrix.timesEquals((double) 1 / 2);
//
//        data = matrix.getArray();
//        for (int i = 0; i < data.length; i++) {
//            for (int j = 0; j < data[0].length; j++) {
//                System.out.print(data[i][j] + " ");
//            }
//            System.out.println("");
//        }
    //   EigenvalueDecomposition ei = new EigenvalueDecomposition(matrix);
//        Matrix result = ei.getD();
//        double[][] out = result.getArray();
//        for(double[] ele : out){
//          for(double e : ele){
//              System.out.print(e+" ");
//          }
//            System.out.println("");
//        }
//        Matrix Vv = ei.getV();
//        double[][] rst = Vv.getArray();
//        for (double[] ele : rst) {
//            for (double e : ele) {
//                System.out.print(e + "   ");
//            }
//            System.out.println("");
//        }
//        double[] rs = ei.getRealEigenvalues();
//        for (double ele : rs) {
//            System.out.println(ele + " ");
//        }
    //System.out.println(matrix.trace());
    //   }
    //Nhan dien anh 
    /**
     * @param imagePath : the path to the test image
     * @param maxDistance :The max distance to be detected
     * @return position or the number name of the detected image.If there's no
     * image match,return -1
     */
    public int imageDetect(String imagePath, double maxDistance) {
        double[] imageWei = getWeight(imagePath);
        double[][] temp = new double[N][N];
        double[] dis = new double[N];
        //Lay cac anh trong co so du lieu tru di anh can nhan dien 
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                temp[i][j] = WW[i][j] - imageWei[j];
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                dis[i] += temp[i][j] * temp[i][j];
            }
            dis[i] = Math.sqrt(dis[i]);

        }

        //Tim anh co khoang cach gan nhat voi anh can nhan dien
        int min = 0;
        for (int i = 1; i < N; i++) {
            if (dis[i] < dis[min]) {
                min = i;
            }
        }
        //Neu khoang cach tu anh co khoang cach nho nhat lon hon maxDistance thi
        //tra lai -1 neu khong thi tra ve vi tri cua anh
        System.out.println("Khoang cach min:" + dis[min]);
        return dis[min] < maxDistance ? min + 1 : -1;
    }

    /**
     *
     * @return nothing but it build a database of set of images in new vector
     * space
     * @param imageSetPath
     *
     */
    public void builtDatabase(String imageSetPath) {
        ImageProcessing java = new ImageProcessing();
        java.getParam(imageSetPath);
        java.buildMeanMatrix(imageSetPath);
        java.buildSymMatrix(imageSetPath);
        java.buildEigenfaceMatrix();
        java.builWeigtOfLearningImages(imageSetPath);

    }

    //Bieu dien cac anh trong khong gian vector dac trung
    private void builWeigtOfLearningImages(String path) {
        WW = new double[N][N];
        for (int i = 0; i < N; i++) {
            String imgPath = path + String.valueOf(i + 1) + ".jpg";
            double[] tmp = getWeight(imgPath);
            for (int j = 0; j < tmp.length; j++) {
                WW[i][j] = tmp[j];
            }
        }
    }

    //Lay ra trong so cua mot buc anh trong khong gian moi
    private double[] getWeight(String imagePath) {
        double[] ww = new double[N];
        double[][] im = new double[wi * he][1];
        //Lay matran diem anh
        Matrix image = getImageMatrix(new File(imagePath));
        //Tru di matran trung binh
        image.minusEquals(meanMatrix);
        //Chuyen ve matran mot chieu
        double[][] temp = image.getArray();
        int row = temp.length;
        int col = temp[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                im[i * col + j][0] = temp[i][j];
            }
        }
        Matrix phi = new Matrix(im);
        Matrix wei = eigenFace.transpose().times(phi);
        double[][] tmp = wei.getArray();
        for (int i = 0; i < N; i++) {
            ww[i] = tmp[i][0];
        }
        return ww;

    }

    //Xay dung eigenface.Ma tran K vector chuan hoa
    private void buildEigenfaceMatrix() {
        EigenvalueDecomposition ei = new EigenvalueDecomposition(symMatrix);
        //Matran eigenFace chua K-so anh luyen cac vector rieng duoc tinh nhu sau
        eigenFace = AA.times(ei.getV());
        //Chuan hoa cac vector ve vector don vi
        double[][] matrix = eigenFace.getArray();
        int M = matrix.length;
        int N = matrix[0].length;
        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int j = 0; j < M; j++) {
                sum += matrix[j][i] * matrix[j][i];
            }
            sum = Math.sqrt(sum);

            for (int j = 0; j < M; j++) {
                matrix[j][i] = matrix[j][i] / sum;
            }
        }
        eigenFace = new Matrix(matrix);

    }

    //Xay dung matran doi xung A^t*A
    private void buildSymMatrix(String path) {
        //Ma tran duoc tao tu cac gia tri diem anh da tru di gia tri trung binh
        //Ki hieu A la ki hieu giong trong tai lieu.
        double[][] A = new double[he * wi][N];
        for (int i = 0; i < N; i++) {
            String imagePath = path + String.valueOf(i + 1) + ".jpg";
            //Lay matran diem anh cua anh
            double tmp[][] = getImageMatrix(new File(imagePath)).getArray();
            //Tru matran diem anh lay duoc tru di matran trung binh
            tmp = new Matrix(tmp).minusEquals(meanMatrix).getArray();
            //gan matran trung binh vao matran A
            for (int j = 0; j < wi; j++) {
                for (int k = 0; k < he; k++) {
                    A[j * he + k][i] = tmp[j][k];
                }
            }
        }
        //Nhan hai ma tran A^t voi A
        AA = new Matrix(A);
        Matrix transpose = AA.transpose();
        symMatrix = transpose.times(AA);
        symMatrix.timesEquals((double) 1 / (N - 1));

    }

    //Xay dung matran trung binh
    private void buildMeanMatrix(String path) {
        //Lay so luong anh trong file anh luyen
        int N = new File(path).listFiles().length;
        String firstImagePath = path + "1.jpg";
        meanMatrix = getImageMatrix(new File(firstImagePath));
        for (int i = 2; i <= N; i++) {
            String imagePath = path + String.valueOf(i) + ".jpg";
            meanMatrix.plusEquals(getImageMatrix(new File(imagePath)));
        }
        meanMatrix.timesEquals((double) 1 / N);

    }

    //Lay matran do sang cua anh
    private Matrix getImageMatrix(File file) {
        try {
            BufferedImage img = ImageIO.read(file);
            Raster raster = img.getData();
            double pixels[][] = new double[wi][he];
            for (int x = 0; x < wi; x++) {
                for (int y = 0; y < he; y++) {
                    pixels[x][y] = raster.getSample(x, y, 0);
                }
            }
            Matrix matrix = new Matrix(pixels);
            return matrix;

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    //Ham lay mot so thong so cua anh
    private void getParam(String path) {
        //Lay so luong anh trong file anh luyen
        N = new File(path).listFiles().length;
        String firstImagePath = path + "1.jpg";
        BufferedImage img;
        try {
            img = ImageIO.read(new File(firstImagePath));
            Raster raster = img.getData();
            wi = raster.getWidth();
            he = raster.getHeight();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

    }

}
