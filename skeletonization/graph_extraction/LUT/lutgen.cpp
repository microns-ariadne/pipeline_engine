#include <iostream>
#include <cmath>
#include <cstdio>
#include <stdlib.h>
#include <list>
#include <omp.h>
#include <queue>
#include <vector>

using namespace std;

const int n = 3;

void Convert(unsigned int val, int ***a)
{
   unsigned int mask = 1 << (27 - 1);
   for(int x = 0; x < n; x++)
   {	
	for(int y=0; y<n; y++) {
	    for(int z=0; z<n; z++) {
   	        int i=x*pow(n,2)+y*n+z;
		if( (val & mask) == 0 )
	       	    a[x][y][z]=0 ;
 	        else
		    a[x][y][z]=1 ;
		mask  >>= 1;
	//	printf("%d",a[x][y][z]);
	     }
	}
    }
}

bool check(int*** a, std::list<int*>  zeros, std::list<int*> ones) {
   bool e=1;
   for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()) && e; it++) {
	int* l=*it;
	if (a[l[0]][l[1]][l[2]]==1)
		e=0;
   }
  
  for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()) && e; it++) {
       	int* l=*it;
	if (a[l[0]][l[1]][l[2]]==0)
	   e=0;
  }

   return e;
}

void add_coord(std::list<int*> &list, char* a, char* b, char* c) {
    
    int xb=a[0]-'0';
    if(a[1]!=':') printf("UNKNOWN COORDINATE");
    int xe=a[2]-'0'+1;

    int yb=b[0]-'0';
    if(b[1]!=':') printf("UNKNOWN COORDINATE");
    int ye=b[2]-'0'+1;

    int zb=c[0]-'0';
    if(c[1]!=':') printf("UNKNOWN COORDINATE");
    int ze=c[2]-'0'+1;


    for(int x=xb; x<xe; x++) {
       for(int y=yb; y<ye; y++) {
         for(int z=zb; z<ze; z++) {
	 	int* temp=(int*)malloc(3*sizeof(int));
		temp[0]=x;
		temp[1]=y;
		temp[2]=z;
		list.push_back(temp);
	}
       }
   }

}

//if inside==true we check whether the inside is remains connected
//o/w we check whether the outside remains connected
bool remainsConnected(int*** a, bool inside) {
   int cubeSide=3;
   int*** visited=(int ***)malloc(cubeSide*sizeof(int**));

   std::vector<int*> inCube;
   std::vector<int*> component;
   std::vector<int*> neighbors;
   std::queue<int*> dfsQueue;

   for (int ix=0; ix<cubeSide; ix++) {
	visited[ix]=(int**)malloc(cubeSide*sizeof(int*));
	for(int iy=0; iy<cubeSide; iy++) {
		visited[ix][iy]=(int*)malloc(cubeSide*sizeof(int));
		for(int iz=0; iz<cubeSide; iz++) {
		    if (a[ix][iy][iz]==inside && (ix!=1 || iy!=1 || iz!=1)) {
		    	int* temp=(int*)malloc(3*sizeof(int));
			temp[0]=ix;
			temp[1]=iy;
			temp[2]=iz;
			neighbors.push_back(temp);
			inCube.push_back(temp);
	             }
	             visited[ix][iy][iz]=false;
		}
          }
  }
	   
   //starting node
   if(neighbors.size()>0) {
      dfsQueue.push(neighbors[0]);
      component.push_back(neighbors[0]);

      int* current =neighbors[0];      

      int current_x = current[0];
      int current_y = current[1];
      int current_z = current[2];

      visited[current_x][current_y][current_z]=true;
   }


   while (!dfsQueue.empty()) {
      int* current = dfsQueue.front();
      dfsQueue.pop();

      int current_x = current[0];
      int current_y = current[1];
      int current_z = current[2];

      for (std::vector<int*>::iterator it = neighbors.begin(); it != neighbors.end(); it++) {
      	int* currentN = *it; 

	int currentN_x = currentN[0];
 	int currentN_y = currentN[1];
     	int currentN_z = currentN[2];
        
        int d_x = currentN_x-current_x;
        int d_y = currentN_y-current_y;
        int d_z = currentN_z-current_z;
	
	if (abs(d_x)<=1 && abs(d_y)<=1 &&  abs(d_z)<=1)
	if (a[currentN_x][currentN_y][currentN_z]==inside)
 	if (!visited[currentN_x][currentN_y][currentN_z]) {
//	       std::cout << d_x << " " << d_y << " " << d_z << " "<< current<<"\n";
               visited[currentN_x][currentN_y][currentN_z]=true;
   //            printf("(%d,%d,%d)->(%d,%d,%d)", current_x, current_y, current_z, currentN_x, currentN_y, currentN_z);
	       dfsQueue.push(currentN);
	       component.push_back(currentN);
	 }
      }
   }

   
// freeing memory
   for (int ix=0; ix<cubeSide; ix++) {
        for(int iy=0; iy<cubeSide; iy++) {
		free(visited[ix][iy]);
        }
	free(visited[ix]);
   }
   free(visited);

   int comps=component.size();
   int cubes=inCube.size();

   for (std::vector<int*>::iterator it = neighbors.begin(); it != neighbors.end(); it++) {int* l=*it; free(l);}
   component.clear();
   inCube.clear();
   neighbors.clear();

   return (comps==cubes);

/* debug
   printf("%d %d\n", comps, cubes);

   for (std::vector<int*>::iterator it = component.begin(); it != component.end(); it++) {
       int* currentN = *it;
       printf("%d %d %d\n",currentN[0],currentN[1],currentN[2]);
   }
   printf("\n");

   for (std::vector<int*>::iterator it = inCube.begin(); it != inCube.end(); it++) {
        int* currentN = *it;
	printf("%d %d %d\n",currentN[0],currentN[1],currentN[2]);
   }*/


//   std::cout << x << " " << y << " " << z <<" "<< (component.size()) << " " <<(inCube.size()) <<"\n";   
//   std::cout << vid << "\n";  
//  if (component.size()!=inCube.size()) {
//    std::cout << x << " " << y << " " << z <<" "<< (component.size()) << " " <<(inCube.size()) <<"\n";   
//  }
//
//   for (int i=0; i<component.size(); i++){
//	printf("%d-%d  ",graph->getVertexData(component[i])->x, graph->getVertexData(component[i])->y);
//   }
//   printf("\n");
//
//  for (int i=0; i<inCube.size(); i++){
//        printf("%d-%d  ",graph->getVertexData(inCube[i])->x, graph->getVertexData(inCube[i])->y);
//   }
//
//   printf("\n");
//
//   }
}

/* int main() {
   int*** cube=Convert(33628304);
   printf("%d\n", cube[0][0][1]);
//   printf("%d",remainsConnected(cube,true));
   return 0;
}
*/

int main()
{
    int N=pow(2,pow(n,3));
    int sum=0;
    int next=0;
    FILE* out = fopen("LUT.txt","w");
    bool* simple = (bool*) malloc (pow(2,pow(n,3))*sizeof(bool));

    int *** a = (int ***)malloc(sizeof(int**) * n);
    for (int x=0; x<n; x++) {
        a[x] = (int **)malloc(sizeof(int *) * n);
	for (int y=0; y<n; y++)
	    a[x][y] = (int *)malloc(sizeof(int) * n);
    }
    for(unsigned int i = 0; i < N; i++) {
      if(i>pow(2,next)) {				          
        next++;
        printf("%d\n",next);
      }

      Convert(i, a);
      simple[i]=1;
      if (remainsConnected(a,"true"))
	   simple[i]=0;

      if (simple[i]==1)
        if (remainsConnected(a,"false"))
          simple[i]=0;
    }
    printf("printing to file!\n");

    for(int i=0;i<N;i++) {
       fprintf(out,"%d",simple[i]);
       if(simple[i]) sum++;
    }
    fclose(out);

    printf("%d\n",sum);
}


int main2()
{
    int N=pow(2,pow(n,3));
    int sum=0;
    int next=0;
    FILE* out = fopen("LUT.txt","w");
    bool* simple = (bool*) malloc (pow(2,pow(n,3))*sizeof(bool));

    int ***a = (int ***)malloc(sizeof(int**) * n);
    for (int x=0; x<n; x++) {
        a[x] = (int **)malloc(sizeof(int *) * n);
	for (int y=0; y<n; y++)
	    a[x][y] = (int *)malloc(sizeof(int) * n);
    }
    
    for(unsigned int i = 0; i < N; i++) {
	if(i>pow(2,next)) {
	   printf("%d of %d\n", pow(2, next), N);
	   next++;
	}

	Convert(i, a);
	std::list<int*> zeros; 
	std::list<int*> ones;
	simple[i]=0;

	zeros.clear();  ones.clear();
	
	
	
	

////////////////// Type A ////////////////////////////////

/////// 1.
        if (simple[i]==0) {  
        add_coord(zeros,"0:0","0:2","0:2");
	add_coord(ones,"1:2","1:1","1:1");     

	if (check(a,zeros,ones)) simple[i]=1;

	for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);}
	zeros.clear();	ones.clear();
	}
/////// 2.
        if (simple[i]==0) {
        add_coord(zeros,"2:2","0:2","0:2");
	add_coord(ones,"0:1","1:1","1:1");     

	if (check(a,zeros,ones)) simple[i]=1;
	
	for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);}
	zeros.clear();	ones.clear();
	}

/////// 3.
        if (simple[i]==0) {
	add_coord(zeros,"0:2","0:0","0:2");
	add_coord(ones,"1:1","1:2","1:1");     

	if (check(a,zeros,ones)) simple[i]=1;
	
	for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
	zeros.clear();	ones.clear();
	}

/////// 4.
        if (simple[i]==0) {
        add_coord(zeros,"0:2","2:2","0:2");
	add_coord(ones,"1:1","0:1","1:1");     

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
/////// 5.
        if (simple[i]==0) {
        add_coord(zeros,"0:2","0:2","0:0");
	add_coord(ones,"1:1","1:1","1:2");     

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
/////// 6.
        if (simple[i]==0) {
        add_coord(zeros,"0:2","0:2","2:2");
	add_coord(ones,"1:1","1:1","0:1");     

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
////////////////////// Type B ////////////////

/////// 1.
        if (simple[i]==0) {
        add_coord(zeros,"0:0","1:2","0:2");
        add_coord(zeros,"1:1","2:2","0:2");
	add_coord(ones,"1:2","1:1","1:1");     
        add_coord(ones,"1:1","0:0","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
/////// 2.
        if (simple[i]==0) {
        add_coord(zeros,"2:2","1:2","0:2");
        add_coord(zeros,"1:1","2:2","0:2");
	add_coord(ones,"0:1","1:1","1:1");     
        add_coord(ones,"1:1","0:0","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
/////// 3.
        if (simple[i]==0) {
        add_coord(zeros,"2:2","0:1","0:2");
        add_coord(zeros,"1:1","0:0","0:2");
	add_coord(ones,"0:1","1:1","1:1");     
        add_coord(ones,"1:1","2:2","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
/////// 4.
        if (simple[i]==0) {
        add_coord(zeros,"0:0","0:1","0:2");
        add_coord(zeros,"1:1","0:0","0:2");
	add_coord(ones,"1:2","1:1","1:1");     
        add_coord(ones,"1:1","2:2","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////// 1'.
        if (simple[i]==0) {
        add_coord(zeros,"0:2","0:0","1:2"); 
        add_coord(zeros,"0:2","1:1","2:2");
	add_coord(ones,"1:1","1:2","1:1");     
        add_coord(ones,"1:1","1:1","0:0");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////// 2'.
        if (simple[i]==0) {
        add_coord(zeros,"0:2","2:2","1:2");
        add_coord(zeros,"0:2","1:1","2:2");
	add_coord(ones,"1:1","0:1","1:1");
        add_coord(ones,"1:1","1:1","0:0");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////// 3'.
        if (simple[i]==0) {
        add_coord(zeros,"0:2","2:2","0:1");
        add_coord(zeros,"0:2","1:1","0:0");
	add_coord(ones,"1:1","0:1","1:1");    
        add_coord(ones,"1:1","1:1","2:2"); 

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
/////// 4'.
        if (simple[i]==0) {
        add_coord(zeros,"0:2","0:0","0:1");
        add_coord(zeros,"0:2","1:1","0:0");
	add_coord(ones,"1:1","1:2","1:1");     
        add_coord(ones,"1:1","1:1","2:2");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}	

/////// 1''.
        if (simple[i]==0) {
        add_coord(zeros,"1:2","0:2","0:0"); 
        add_coord(zeros,"2:2","0:2","1:1");
	add_coord(ones,"1:1","1:1","1:2");     
        add_coord(ones,"0:0","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////// 2''.
        if (simple[i]==0) {
        add_coord(zeros,"1:2","0:2","2:2");
        add_coord(zeros,"2:2","0:2","1:1");
	add_coord(ones,"1:1","1:1","0:1");
        add_coord(ones,"0:0","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////// 3''.
        if (simple[i]==0) {
        add_coord(zeros,"0:1","0:2","2:2");
        add_coord(zeros,"0:0","0:2","1:1");
	add_coord(ones,"1:1","1:1","0:1");    
        add_coord(ones,"2:2","1:1","1:1"); 

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////// 4''.
        if (simple[i]==0) {
        add_coord(zeros,"0:1","0:2","0:0");
        add_coord(zeros,"0:0","0:2","1:1");
	add_coord(ones,"1:1","1:1","1:2");     
        add_coord(ones,"2:2","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////////////////////// Type C //////////////////////////

///////// 1.
        if (simple[i]==0) {
	add_coord(zeros,"1:2","0:0","0:1");
        add_coord(zeros,"2:2","1:1","0:1");
	add_coord(zeros,"1:1","1:1","0:0");
	add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"0:0","1:1","1:1");
        add_coord(ones,"1:1","2:2","1:1");
        add_coord(ones,"1:1","1:1","2:2");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 2.
        if (simple[i]==0) {
        add_coord(zeros,"0:1","0:0","0:1");
        add_coord(zeros,"0:0","1:1","0:1");
	add_coord(zeros,"1:1","1:1","0:0");
	add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"2:2","1:1","1:1");
        add_coord(ones,"1:1","2:2","1:1");
        add_coord(ones,"1:1","1:1","2:2");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
///////// 3.
        if (simple[i]==0) {
        add_coord(zeros,"0:1","2:2","0:1");
        add_coord(zeros,"0:0","1:1","0:1");
	add_coord(zeros,"1:1","1:1","0:0");
	add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"2:2","1:1","1:1");
        add_coord(ones,"1:1","0:0","1:1");
        add_coord(ones,"1:1","1:1","2:2");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 4
        if (simple[i]==0) {
        add_coord(zeros,"1:2","2:2","0:1");
        add_coord(zeros,"2:2","1:1","0:1");
	add_coord(zeros,"1:1","1:1","0:0");
	add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"0:0","1:1","1:1");
        add_coord(ones,"1:1","0:0","1:1");
        add_coord(ones,"1:1","1:1","2:2");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 5.
        if (simple[i]==0) {
        add_coord(zeros,"1:2","0:0","1:2");
        add_coord(zeros,"2:2","1:1","1:2");
	add_coord(zeros,"1:1","1:1","2:2");
	add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"0:0","1:1","1:1");
        add_coord(ones,"1:1","2:2","1:1");
        add_coord(ones,"1:1","1:1","0:0");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 6.
        if (simple[i]==0) {
        add_coord(zeros,"0:1","0:0","1:2");
        add_coord(zeros,"0:0","1:1","1:2");
	add_coord(zeros,"1:1","1:1","2:2");
	add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"2:2","1:1","1:1");
        add_coord(ones,"1:1","2:2","1:1");
        add_coord(ones,"1:1","1:1","0:0");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
///////// 7.
        if (simple[i]==0) {
        add_coord(zeros,"0:1","2:2","1:2");
        add_coord(zeros,"0:0","1:1","1:2");
	add_coord(zeros,"1:1","1:1","2:2");
	add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"2:2","1:1","1:1");
        add_coord(ones,"1:1","0:0","1:1");
        add_coord(ones,"1:1","1:1","0:0");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}
///////// 8.
        if (simple[i]==0) {
        add_coord(zeros,"1:2","2:2","1:2");    	
        add_coord(zeros,"2:2","1:1","1:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(ones,"1:1","1:1","1:1");     
        add_coord(ones,"0:0","1:1","1:1");
        add_coord(ones,"1:1","0:0","1:1");
        add_coord(ones,"1:1","1:1","0:0");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

/////////////////// Type D
///////// 1.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(zeros,"0:0","1:1","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(zeros,"2:2","1:1","2:2");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(zeros,"2:2","1:1","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(ones,"0:0","1:1","0:0");     
        add_coord(ones,"1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 2.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(zeros,"0:0","1:1","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(zeros,"2:2","1:1","2:2");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(ones,"2:2","1:1","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(zeros,"0:0","1:1","0:0");     
        add_coord(ones,"1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 3.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(zeros,"0:0","1:1","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(ones,"2:2","1:1","2:2");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(zeros,"2:2","1:1","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(zeros,"0:0","1:1","0:0");     
        add_coord(ones,"1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 4.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(ones,"0:0","1:1","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(zeros,"2:2","1:1","2:2");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(zeros,"2:2","1:1","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(zeros,"0:0","1:1","0:0");     
        add_coord(ones,"1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 1'.
        if (simple[i]==0) {
	add_coord(zeros,"1:1","0:0","1:1");    	
        add_coord(zeros,"1:1","0:0","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(zeros,"1:1","2:2","2:2");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(zeros,"1:1","2:2","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(ones, "1:1","0:0","0:0");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 2'.
        if (simple[i]==0) {
	add_coord(zeros,"1:1","0:0","1:1");    	
        add_coord(zeros,"1:1","0:0","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(zeros,"1:1","2:2","2:2");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(ones, "1:1","2:2","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(zeros,"1:1","0:0","0:0");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 3'.
        if (simple[i]==0) {
	add_coord(zeros,"1:1","0:0","1:1");    	
        add_coord(zeros,"1:1","0:0","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(ones, "1:1","2:2","2:2");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(zeros,"1:1","2:2","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(zeros,"1:1","0:0","0:0");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 4'.
        if (simple[i]==0) {
	add_coord(zeros,"1:1","0:0","1:1");    	
        add_coord(ones, "1:1","0:0","2:2");
        add_coord(zeros,"1:1","1:1","2:2");
        add_coord(zeros,"1:1","2:2","2:2");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(zeros,"1:1","2:2","0:0");
        add_coord(zeros,"1:1","1:1","0:0");
	add_coord(zeros,"1:1","0:0","0:0");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 1''.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(zeros,"0:0","2:2","1:1");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(zeros,"2:2","2:2","1:1");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(zeros,"2:2","0:0","1:1");
        add_coord(zeros,"1:1","0:0","1:1");
	add_coord(ones, "0:0","0:0","1:1");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 2''.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(zeros,"0:0","2:2","1:1");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(zeros,"2:2","2:2","1:1");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(ones, "2:2","0:0","1:1");
        add_coord(zeros,"1:1","0:0","1:1");
	add_coord(zeros,"0:0","0:0","1:1");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 3''.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(zeros,"0:0","2:2","1:1");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(ones, "2:2","2:2","1:1");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(zeros,"2:2","0:0","1:1");
        add_coord(zeros,"1:1","0:0","1:1");
	add_coord(zeros,"0:0","0:0","1:1");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

///////// 4''.
        if (simple[i]==0) {
	add_coord(zeros,"0:0","1:1","1:1");    	
        add_coord(ones, "0:0","2:2","1:1");
        add_coord(zeros,"1:1","2:2","1:1");
        add_coord(zeros,"2:2","2:2","1:1");
        add_coord(zeros,"2:2","1:1","1:1");
        add_coord(zeros,"2:2","0:0","1:1");
        add_coord(zeros,"1:1","0:0","1:1");
	add_coord(zeros,"0:0","0:0","1:1");     
        add_coord(ones, "1:1","1:1","1:1");

	if (check(a,zeros,ones)) simple[i]=1;
        
        for (std::list<int*>::iterator it = zeros.begin(); (it != zeros.end()); it++) {int* l=*it; free(l);}
        for (std::list<int*>::iterator it = ones.begin(); (it != ones.end()); it++) { int* l=*it; free(l);} 	
        zeros.clear();	ones.clear();
	}

//Connectedness
        if (simple[i]==1) 
		if (!remainsConnected(a,"true"))
			simple[i]=0;

        if (simple[i]==1) 
	        if (!remainsConnected(a,"false"))
		        simple[i]=0;


/*Testing
	printf("%d %d: ",simple[i],i);
        for(int x = 0; x < n; x++)
        {	
       	   for(int y=0; y<n; y++) {
	       for(int z=0; z<n; z++) {   
		    printf("%d",a[x][y][z]) ;
	       }
           }
        }
        printf("\n");*/




/// Cleanup
        for (int ix=0; ix<n; ix++) {
            for(int iy=0; iy<n; iy++) {
	        free(a[ix][iy]);
	    }
	    free(a[ix]);
	}
	free(a);

    }
    printf("printing to file!\n");
    for(int i=0;i<N;i++) {
	fprintf(out,"%d",simple[i]);
	if(simple[i]) sum++;
    }
    fclose(out);
    
    printf("%d\n",sum);


/*    bool* simple[i] = (bool*) malloc (pow(2,pow(n,3))*sizeof(bool));
    int*** cube=Convert(6);
    FILE* in = fopen("LUT.txt","r");
    for(int i=0; i<N; i++) {

   	char temp;
	fscanf(in,"%c",&temp);
	if (temp=='0')
    		simple[i][i]=0;
	else if (temp=='1')
		simple[i][i]=1;
	else
		printf("ERROR, %d",temp);

	sum+=simple[i][i];
    }

    printf("%d\n",sum);


/*    int ex=pow(n,3)-1; //exponent
    int val=0; //hash value
    for(int x=0; x < n; x++) {	
        for(int y=0; y<n; y++) {
              for(int z=0; z<n; z++) {
	            val+=cube[x][y][z]*pow(2,ex);
		    ex--;
               }
         }
    }

    printf("%d\n",simple[i][val]);
*/
}
