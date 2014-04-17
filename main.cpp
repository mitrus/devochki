#include <opencv2\opencv.hpp>
#include <cmath>
#include <queue>

#define GRAY(R, G, B) (.299 * R + .587 * G + .144 * B)

using namespace cv;
using namespace std;

const string pathToPhotos = "C:\\photos\\";
const int countOfPhotos = 100;
const double possible_bfs_distance = 50;
const int ignore_border = 60;
const double segment_coeff = 0.3;
const double step = 0.05;
const double min_angle = 3.1416 * 50 / 180.0;
const double max_angle = 3.1416 * 130 / 180.0;

/** Global variables */
String face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

inline double sqr(double a) { return a * a; }

double dist3d(Vec3b a, Vec3b b) {
	return sqrt((sqr((int) a[0] - (int) b[0]) + sqr((int) a[1] - (int) b[1]) + sqr((int) a[2] - (int) b[2])) / 3.0);
}

Vec3b getSkinColor(Mat &pic, Rect forehead) {
	int answer[3] = {};
	int count = 0;
	for (int i = forehead.x; i < forehead.x + forehead.width; i++)
		for (int j = forehead.y; j < forehead.y + forehead.height; j++) {
			if (GRAY(pic.at<Vec3b>(j, i)[2], pic.at<Vec3b>(j, i)[1], pic.at<Vec3b>(j, i)[0]) > ignore_border) {
				answer[0] += pic.at<Vec3b>(j, i)[0];
				answer[1] += pic.at<Vec3b>(j, i)[1];
				answer[2] += pic.at<Vec3b>(j, i)[2];
				count++;
			}
		}
	for (int i = 0; i < 3; i++) answer[i] /= count;
	return Vec3b(answer[0], answer[1], answer[2]);
}

vector< vector<bool> > used;
int vx[4] = {0, 0, 1, -1},
	vy[4] = {1, -1, 0, 0};

bool isBlue(Vec3b a) {
	return a[0] == 255 && a[1] + a[2] == 0;
}

bool isRed(Vec3b a) {
	return (int) a[2] == 255 && (int) a[0] == 0 && (int) a[1] == 0;
}

void bfsFromPoint(Mat &pic, Point p, Vec3b face_color, int max_x) {
	queue<Point> q;
	q.push(p);
	used[p.x][p.y] = 1;
	while (!q.empty()) {
		Point cur_point = q.front();
		q.pop();
		pic.at<Vec3b>(cur_point.x, cur_point.y) = Vec3b(0, 0, 255);
		for (int i = 0; i < 4; i++) {
			int x = cur_point.x + vx[i],
				y = cur_point.y + vy[i];
			if (x >= 0 && y >= 0 && x < min(pic.rows, max_x) && y < pic.cols && !used[x][y] && dist3d(face_color, pic.at<Vec3b>(x, y)) < possible_bfs_distance) {
				used[x][y] = 1;
				q.push(Point(x, y));
			}
		}
	}
}

void bfsOnFace(Mat &pic, Rect forehead, Vec3b face_color) {
	used.clear();
	used.resize(pic.rows, vector<bool>(pic.cols, false));
	for (int i = 0; i < forehead.width; i++)
		for (int j = 0; j < forehead.height; j++) {
			int x = forehead.x + i,
				y = forehead.y + j;
			swap(x, y);
			bfsFromPoint(pic, Point(x, y), face_color, forehead.y + forehead.height);
		}
}

inline double dist2p(Point a, Point b) {
	return sqrt(sqr(a.x - b.x) + sqr(a.y - b.y));
}

vector< pair<Vec3b, double> > getSegment(Mat &pic, Point center, double angle, double length) {
	vector< pair<Vec3b, double> > tmp;
	for (int i = center.x - 1; i >= 0; i--) {
		int dy = (center.x - i) / tan(angle);
		int xx = i,
			yy = center.y + dy;
		Point p(xx, yy);
		if (dist2p(p, center) > length) break;
		if (isRed(pic.at<Vec3b>(xx, yy)) || isBlue(pic.at<Vec3b>(xx, yy))) continue;
		tmp.push_back(make_pair(pic.at<Vec3b>(xx, yy), .0));
		pic.at<Vec3b>(xx, yy) = Vec3b(255, 0, 0);
	}
	double n = tmp.size();
	for (int i = 0; i < (int) tmp.size(); i++) 
		tmp[i].second = ((double) tmp.size() - i) / n;
	double summary = 0;
	for (int i = 0; i < (int) tmp.size(); i++)
		summary += tmp[i].second;
	for (int i = 0; i < (int) tmp.size(); i++) 
		tmp[i].second /= summary;
	return tmp;
}

void proceedPhoto(Mat &pic) {
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(pic, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faces.size(); i++) {
		Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		Rect forehead;
		forehead.height = faces[i].height / 5.0;
		forehead.y = faces[i].y;
		forehead.x = faces[i].x + 2 * faces[i].width / 5.0;
		forehead.width = faces[i].width / 5.0;
//		rectangle(pic, forehead, Scalar(255, 0, 255));
		Vec3b skin = getSkinColor(pic, forehead);
		bfsOnFace(pic, forehead, skin);
		

		Point middle_point = Point(forehead.y + forehead.height / 2.0, forehead.x + forehead.width / 2.0);
		vector< pair<Vec3b, double> > total;
		for (double angle = min_angle; angle < max_angle; angle += step) {
			vector< pair<Vec3b, double> > tmp = getSegment(pic, middle_point, angle, faces[i].height * segment_coeff);
			for (int j = 0; j < (int) tmp.size(); j++)
				total.push_back(tmp[j]);
		}
		double summary = 0;
		for (int j = 0; j < (int) total.size(); j++) 
			summary += total[j].second;
		for (int j = 0; j < (int) total.size(); j++)
			total[j].second /= summary;
		double COLOR[3] = {};
		for (int j = 0; j < (int) total.size(); j++) 
			for (int k = 0; k < 3; k++)
				COLOR[k] += total[j].first[k] * total[j].second;
		Vec3b result_color(COLOR[0], COLOR[1], COLOR[2]);
		for (int x = 0; x <= 20; x++)
			for (int y = 0; y <= 20; y++)
				pic.at<Vec3b>(x, y) = result_color;
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		
		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for (size_t j = 0; j < eyes.size(); j++) {
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5, faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
		//	circle(pic, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	//-- Show what you got
	imshow("LOL", pic);
	waitKey();
}

/** @function main */
int main( int argc, const char** argv ) {
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	for (int i = 2; i <= countOfPhotos; i++) {
		char number[20];
		sprintf(number, "%d", i);
		string num_str(number);
		string current_photo_name = pathToPhotos + num_str + ".jpg";
		Mat pic = imread(current_photo_name.c_str());
		if (pic.data)
			proceedPhoto(pic);
	}
}
