
library(ff)
library(ggplot2)
library(ggpubr)
library(ggrepel)

# Load tables

cercacovid_data= "CERCACOVID_questionari_clean_2020-05-04.csv" 
score_data = "CERCACOVID_questionari_score.csv"

x <- read.csv.ffdf(file= cercacovid_data, header=TRUE, VERBOSE=TRUE, first.rows=10000, next.rows=50000, colClasses=NA)
df <- as.data.frame(x)

score <- read.csv.ffdf(file= score_data, header=TRUE, VERBOSE=TRUE, first.rows=10000, next.rows=50000, colClasses=NA)
df_score <- as.data.frame(score)

tamp <- read.csv("dpc-covid19-ita-regioni.csv")

mort <- read.delim("Mortality_ISTAT.tab")

# Format data

df <- cbind.data.frame(df, df_score$SCORE)
colnames(df)[ncol(df)] <- "SCORE"
df$DT_CREATION_DATA <- as.Date(df$DT_CREATION_DATA)

##################
##### FIGURE 1
##################

df_ord <- df[order(df$DT_CREATION_DATA, decreasing=T),]
df_ord_uni <- df_ord[!duplicated(df_ord$ID_SURVEY_DETAIL_ANAG),]

FEVER <- df_ord_uni$FEBBRE > 37.5

df2 <- cbind.data.frame(df_ord_uni[,25:31])

df3 <- apply(df2, 2, function(x) grepl("Si", x))

df4 <- cbind.data.frame(FEVER, df3)

data <- df4

mat <- matrix(nrow=0, ncol=8)
colnames(mat) <- c("col1","col2","p.value","odd_ratio","c1F_c2F","c1F_c2T","c1T_c2F","c1T_c2T")

for (i in 1:(ncol(data)-1))
{
	col1 <- as.matrix(colnames(data)[i])

	for (u in (i+1):(ncol(data)))
	{
		col2 <- as.matrix(colnames(data)[u])

  		test <- fisher.test(data[,i], data[,u])

		odd_ratio <- as.numeric(test$estimate)
			
		tt <- as.matrix(table(data[,i], data[,u]))

		#FALSE FALSE
		num_11 <- tt[1,1]

		#column u is TRUE, column i is FALSE
		num_12 <- tt[1,2]

		#column i is TRUE, column u is FALSE
		num_21 <- tt[2,1]

		# both TRUE
		num_22 <- tt[2,2]

		tmp <- cbind.data.frame(col1, col2, test$p.value, odd_ratio, num_11, num_12, num_21, num_22)
		colnames(tmp) <- colnames(mat)

		mat <- rbind(mat, tmp)
	}

}

mat <- as.data.frame(mat)

# HEATMAP p(AB)/(p(A)*p(B))

m_mat <- matrix(ncol=ncol(data), nrow=ncol(data))
row.names(m_mat) <- colnames(data)
colnames(m_mat) <- colnames(data)

for (i in 1:nrow(mat))
{
	C1 <- sum(df4[,as.matrix(mat[i,"col1"])])/nrow(df4)
	C2 <- sum(df4[,as.matrix(mat[i,"col2"])])/nrow(df4)
	C1C2 <- as.numeric(mat[i,"c1T_c2T"])/(sum(mat[i, c("c1F_c2F","c1F_c2T","c1T_c2F","c1T_c2T")]))

	m_mat[as.matrix(mat[i, "col1"]), as.matrix(mat[i, "col2"])] <- C1C2/(C1*C2)
	m_mat[as.matrix(mat[i, "col2"]), as.matrix(mat[i, "col1"])] <- m_mat[as.matrix(mat[i, "col1"]), as.matrix(mat[i, "col2"])]
}

colnames(m_mat) <- c("Fever","Disgeusia/Ageusia","Cough","Muscle pain","Fatigue","Conjunctivitis","Diarrhoea","Nasal obstruction")
row.names(m_mat) <- c("Fever","Disgeusia/Ageusia","Cough","Muscle pain","Fatigue","Conjunctivitis","Diarrhoea","Nasal obstruction")

my_palette <- colorRampPalette(c("darkgreen","black","yellow","yellow","orange","orange","red","red"))(n = 150)

library(gplots)

pdf("Figure1.pdf")
heatmap.2(m_mat, trace="none", col = my_palette, cexRow=1.3, cexCol=1.4, srtCol = 45, margins=c(11,11), key.title="", lhei = c(10,28), key.par = list(cex.axis=1.2, cex.lab=0.00001))
dev.off()

########################################
##### FIGURE 2 e EXTENDED DATA FIGURE 8
########################################

df2 <- df

colnames(df2)[colnames(df2) == "DT_CREATION_DATA"] <- "Date"

Possibly_Covid <- (df2$SCORE >= 8)*1

df2 <- cbind.data.frame(df2, Possibly_Covid)

df2 <- df2[df2$Date >= as.Date("2020-04-01"),]

freq_PutativeCovid <- aggregate(df2$Possibly_Covid, list(df2$Date), mean)

colnames(freq_PutativeCovid) <- c("data","Freq_PutativeCovid")

tamp_lomb <- subset(tamp, tamp$denominazione_regione == "Lombardia")

var_tamp <- matrix(ncol=2, nrow=0)
colnames(var_tamp) <- c("data","variazione_tamponi")

for (i in 2:nrow(tamp_lomb))
{
	var <- tamp_lomb[i,"tamponi"]-tamp_lomb[i-1,"tamponi"]

	tmp <- cbind.data.frame(tamp_lomb[i,"data"], var)
	colnames(tmp) <- colnames(var_tamp)
	var_tamp <- rbind(var_tamp, tmp)

}

tamp_lomb2 <- merge(tamp_lomb, var_tamp, by="data", all.x = T)

freq_tamponi_positivi <- tamp_lomb2$variazione_totale_positivi/tamp_lomb2$variazione_tamponi

tamp_lomb3 <- cbind.data.frame(tamp_lomb2, freq_tamponi_positivi)

tamp_lomb3$data <- as.Date(tamp_lomb3$data)
 
tot <- merge(tamp_lomb3, freq_PutativeCovid, by="data", all.x = T)

freq_new_pos <- tot$nuovi_positivi/tot$variazione_tamponi

tot3 <- cbind.data.frame(tot,freq_new_pos)

tot2 <- tot3[!is.na(tot3$Freq_PutativeCovid),]

coeff = 0.00007

g <- ggplot(data=tot3, aes(x=data), size = 7) + geom_point(aes(y = Freq_PutativeCovid/coeff, colour = "Frequency of Putative COVID + users")) + geom_point(aes(y = nuovi_positivi, colour = "Number of swab-positive subjects")) +  geom_line(aes(y = Freq_PutativeCovid/coeff, colour = "Frequency of Putative COVID + users")) + geom_line(aes(y = nuovi_positivi, colour = "Number of swab-positive subjects")) + ylab("Frequency") + xlab("Date") + theme(text = element_text(size = 35)) +  theme_bw() + scale_y_continuous(name = "Number of swab-positive subjects",sec.axis = sec_axis(~.*coeff, name="Frequency of Putative COVID + users")) + labs(color='', size=10) + theme(legend.position = 'bottom',axis.title.y = element_text(size = 16), legend.text=element_text(size=12))

ggsave(g, file = "Figure2.pdf")

g <- ggplot(data=tot2, aes(x=nuovi_positivi, y=Freq_PutativeCovid)) + geom_smooth(method='lm', color="red", se=T) + geom_point(color="red", size=4)  + stat_cor(method = "pearson", size=5) + theme(text = element_text(size = 25)) +  theme_bw() + xlab("Number of swab-positive subjects") + ylab("Frequency of Putative COVID + users") + theme(legend.position = 'bottom',axis.title.y = element_text(size = 16), axis.title.x = element_text(size = 16), legend.text=element_text(size=12))

ggsave(g, file="Extended_Data_Figure8.pdf")

################
## FIGURE 3
################


df2_1apr <- subset(df2, df2$Date == as.Date("2020-04-01"))

freq_1apr <- aggregate(df2_1apr$Possibly_Covid, list(df2_1apr$DOMICILIO_PROVINCIA), mean)
colnames(freq_1apr) <- c("Provincia_sigla","PutativeCovidfreq")

m <- merge(mort, freq_1apr, by="Provincia_sigla")
m <- as.data.frame(m)

g <- ggplot(data=m, aes(x=Decessi_Covid.pop.100, y=PutativeCovidfreq, label = Provincia_sigla)) + geom_point(color="red", size=4) + geom_smooth(method='lm', color="red") + stat_cor(method = "pearson", size=5) + geom_label_repel(aes(label = Provincia_sigla), box.padding   = 0.35, point.padding = 0.6, segment.color = 'grey50') + ylab("Frequency of Putative COVID + users") + xlab("COVID-19 mortality") + theme(text = element_text(size = 25)) +  theme_bw()

ggsave(g, file="Figure3.pdf")


##########################
## Extended Data Figure 1
##########################

pdf("Extended_Data_Figure1.pdf", 15, 15)
hist(df$DT_CREATION_DATA, breaks=100, col="forestgreen", las =2, freq=T, xlab="", ylab="", main = "")
abline(v = as.Date("2020-04-16"), col="forestgreen", lty=5)
abline(v = as.Date("2020-04-21"), col="forestgreen", lty=5)
abline(v = as.Date("2020-04-01"), col="forestgreen", lty=5)
dev.off()

df_prima <- df[df$DT_CREATION_DATA < as.Date("2020-04-16"),]
df_dopo <- df[df$DT_CREATION_DATA >= as.Date("2020-04-16"),]

median(table(df_prima$DT_CREATION_DATA))
#[1] 54793

median(table(df_dopo$DT_CREATION_DATA))
#[1] 87052.5

before <- as.matrix(table(df_prima$DT_CREATION_DATA))[,1]
after <- as.matrix(table(df_dopo$DT_CREATION_DATA))[,1]

wilcox.test(before, after, exact=F)

#	Wilcoxon rank sum test with continuity correction

#data:  before and after
#W = 71, p-value = 0.007147
#alternative hypothesis: true location shift is not equal to 0


##########################
## Extended Data Figure 2
##########################

df_ord <- df[order(df$DT_CREATION_DATA),]
df_ord_uni <- df_ord[!duplicated(df_ord$ID_SURVEY_DETAIL_ANAG),]

pdf("Extended_Data_Figure2.pdf", 15, 15)
hist(df_ord_uni$DT_CREATION_DATA, breaks=100, col="dodgerblue2", las =2, freq=T, xlab="", ylab="", main = "")
abline(v = as.Date("2020-04-16"), col="dodgerblue4", lty=5)
abline(v = as.Date("2020-04-21"), col="dodgerblue4", lty=5)
abline(v = as.Date("2020-04-01"), col="dodgerblue4", lty=5)
dev.off()

##################################
## USERS WITH MORE QUESTIONNAIRES
##################################

t <- as.matrix(table(df$ID_SURVEY_DETAIL_ANAG, df$DT_CREATION_DATA))
multi <- row.names(t)[rowSums(t) > 1]

# maximum numer of questionnaires
max(rowSums(t))
#[1] 34

df_multi <- df[df$ID_SURVEY_DETAIL_ANAG %in% multi,]
df_multi_ord <- df_multi[order(df_multi$DT_CREATION_DATA),]
df_multi_ord_uni <- df_multi_ord[!duplicated(df_multi_ord$ID_SURVEY_DETAIL_ANAG),]

df_nomulti <- df[!(df$ID_SURVEY_DETAIL_ANAG %in% multi),]
df_nomulti_ord <- df_nomulti[order(df_nomulti$DT_CREATION_DATA),]
df_nomulti_ord_uni <- df_nomulti_ord[!duplicated(df_nomulti_ord$ID_SURVEY_DETAIL_ANAG),]

# COMPARE SINGLE AND MULTIPLE QUSTIONNAIRE USERS

t_multi_users_uni <- as.matrix(table(df_multi_ord_uni$DT_CREATION_DATA))
t_nomulti_users_uni <- as.matrix(table(df_nomulti_ord_uni$DT_CREATION_DATA))

m <- merge(t_nomulti_users_uni, t_multi_users_uni, by="row.names")
colnames(m) <- c("Data","Users_NOmulti_quest", "Users_multi_quest")

cor.test(m$Users_multi_quest,m$Users_NOmulti_quest, method = "spearman")

#	Spearman's rank correlation rho

#data:  m$Users_multi_quest and m$Users_NOmulti_quest
#S = 634, p-value < 2.2e-16
#alternative hypothesis: true rho is not equal to 0
#sample estimates:
#      rho 
#0.9031322 

##########################
## Extended Data Figure 7
##########################

df2_a <- aggregate(df2$Possibly_Covid, list(df2$Date), mean)
colnames(df2_a) <- c("Date","Possibly_Covid")

g2 <- ggplot(data=df2_a, aes(x=Date, y=Possibly_Covid)) + geom_smooth(method='lm', color="red", size=3) + stat_summary(geom = "point", color="gray40", size = 8) + theme_bw() + theme(strip.text = element_text(size=23)) + geom_vline(xintercept = as.Date("2020-04-16"), linetype="dotted",color = "gray30", size=2) + geom_vline(xintercept = as.Date("2020-04-21"), linetype="dotted",color = "gray30", size=2) + theme(text = element_text(size = 25)) + ylab("Frequency of Putative COVID + users") + theme(axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 20)) + stat_cor(method = "pearson", label.y = 0.01, color="red", size=15)

ggsave(g2, file="Extended_Data_Figure7.pdf", width = 12, height = 12)

##########################
## Extended Data Figure 9
##########################

df2_a <- aggregate(df2$Possibly_Covid, list(df2$Date, df2$DOMICILIO_PROVINCIA), mean)
colnames(df2_a) <- c("Date","DOMICILIO_PROVINCIA","Possibly_Covid")

g <- ggplot(data=df2_a, aes(x=Date, y=Possibly_Covid), group = DOMICILIO_PROVINCIA) + geom_smooth(method='lm', color="red") + stat_summary(geom = "point", fun = mean, color="gray40") + theme_bw() + facet_wrap(~DOMICILIO_PROVINCIA,  ncol=3) + theme(strip.text = element_text(size=23)) + geom_vline(xintercept = as.Date("2020-04-16"), linetype="dotted",color = "gray30", size=1.5) + geom_vline(xintercept = as.Date("2020-04-21"), linetype="dotted",color = "gray30", size=1.5) + stat_cor(method = "pearson", label.y = 0.01, color="red", size=5) + ylab("Frequency of Putative COVID + users") + theme(axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 20))

ggsave(g, file="Extended_Data_Figure9.pdf", width = 12, height = 12)


