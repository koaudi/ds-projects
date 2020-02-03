DROP TEMPORARY TABLE coupons;

CREATE TEMPORARY TABLE coupons (
    coupon_text VARCHAR(255)
);



INSERT INTO 
   coupons (coupon_text)
VALUES
     ('HAPPY123',),
	 ('H129183AA',),
	 ('SPRINGSALE',),
	 ('SPRINGSAVINGS',),
	 ('LACE008'),
	 ('NEW20OFF');


select max(coupon_text) from coupons

MAX(coupon_text) will select SPRINGSAVINGS since the coupon_text column is a string the max function will sort the rows alphabetically asceding and pick the last rows
