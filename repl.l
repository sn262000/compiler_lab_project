%{
#include<stdio.h>
#include<string.h>
char p[20],q[20],r[20],fn[20];
%}


%%

[a-zA-Z]+ {
                if(strcmp(p,yytext)==0)
                        fprintf(yyout,"%s",q);
                else
                        fprintf(yyout,"%s",yytext);

        }
\n   {fprintf(yyout,"%s",yytext);}

.       {fprintf(yyout,"%s",yytext);}
%%
int main(int argc, char *argv[])
{
        strcpy(fn,argv[1]);
        strcpy(p,argv[2]);
        strcpy(q,argv[3]);
        yyin=fopen(fn,"r");    
        yyout=fopen("copy.txt","w");
        yylex();
        return(0);
}