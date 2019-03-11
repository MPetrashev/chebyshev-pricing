library(chebpol)
dims <- c(x=101)
f <- function(x) x*x + 2
#value <- evalongrid(f , dims)
#ch <- ipol(f,method='cheb')
ch <- ipol(f,dims=10,method='cheb')
print(ch(5))
evalongrid(f,11,intervals=c(-1,1))

knots <- expand.grid( chebknots(11,intervals=c(-1,1)) )
chebcoef( evalongrid(f,11,intervals=c(-1,1)) )

g <- function(x) x[1]  + x[2]*10
evalongrid(g,c(5,5),intervals=list(c(-1,1),c(-1,1)))

ipol(g,dims=c(11,11),grid=c(chebknots(11),chebknots(11)),method='cheb')
ipol(g,grid=c(chebknots(11),chebknots(11)),method='cheb')

v <- expand.grid(c(1,2),c(3,4))

grid <- merge(chebknots(11)[[1]],chebknots(11)[[1]])
ipol(g(grid['x'],grid['y']),grid=c(chebknots(11),chebknots(11)),method='cheb')


make.g <- function(){
    count <- 0
    g1 <- function(x){
        count <- count + 1
        count + x
    }
    g1
}

